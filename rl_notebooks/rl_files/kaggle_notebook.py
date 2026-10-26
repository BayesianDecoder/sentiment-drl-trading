# =============================================================
# KAGGLE NOTEBOOK — Sentiment-Augmented DRL Trading System
# Account: any | GPU: T4 x1 | Expected: ~6-8 hrs
# =============================================================
# BEFORE RUNNING:
#  1. Add dataset: vijayadi/sentiment-stocks (SEntFiN-v1.1.csv)
#  2. Kaggle Secrets → HF_TOKEN
#  3. GPU: T4 x1 | Internet: ON
# =============================================================

# %% CELL 1 — Install
# !pip install -q -U transformers peft trl datasets yfinance gymnasium

# %% CELL 2 — Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import yfinance as yf
import json, os, time, gc, copy, re, hashlib, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# HF Login
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(token=UserSecretsClient().get_secret("HF_TOKEN"))
    print("✅ HF login OK")
except Exception as e:
    print(f"⚠️  HF login: {e}")

# %% CELL 3 — Configuration
ASSET       = "TSLA"             # change to "BTC-USD" for Bitcoin
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2024-07-31"
TEST_START  = "2024-08-01"
MAX_EPISODES  = 1000             # reduce for faster testing
EVAL_EVERY    = 50
PATIENCE      = 10
INITIAL_VALUE = 100_000.0
WARMUP_DAYS   = 60

CSV_PATH  = "/kaggle/input/datasets/vijayadi/sentiment-stocks/SEntFiN-v1.1.csv"
SAVE_DIR  = "/kaggle/working"
Path(SAVE_DIR).mkdir(exist_ok=True)

# %% CELL 4 — Data Loading
def load_prices(ticker: str) -> pd.DataFrame:
    print(f"Downloading {ticker} prices...")
    df = yf.download(ticker, start="2010-01-01", auto_adjust=True)
    df = df[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.index = pd.to_datetime(df.index).normalize()
    return df.dropna().reset_index().rename(columns={"Date":"date", "index":"date"})

prices_df = load_prices(ASSET)
prices_df["date"] = pd.to_datetime(prices_df["date"])
print(f"✅ {len(prices_df)} trading days | {prices_df['date'].min().date()} → {prices_df['date'].max().date()}")

# Splits
train_df = prices_df[prices_df["date"] <= TRAIN_END].copy()
val_df   = prices_df[(prices_df["date"] >= VAL_START) & (prices_df["date"] <= VAL_END)].copy()
test_df  = prices_df[prices_df["date"] >= TEST_START].copy()
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %% CELL 5 — Sentiment Scoring
print("Loading SEntFiN sentiment model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO = "vijay5799760/Qwen2.5-3B-SEntFiN-LoRA-HPtuned"
BASE = "Qwen/Qwen2.5-3B-Instruct"

sent_tok = AutoTokenizer.from_pretrained(REPO)
sent_base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="auto"
)
sent_model = PeftModel.from_pretrained(sent_base, REPO)
sent_model.eval()

# Pre-compute label token IDs
label_ids = {}
for label in ["positive", "negative", "neutral"]:
    variants = [label, " "+label, label.capitalize(), " "+label.capitalize()]
    ids = []
    for v in variants:
        ids.extend(sent_tok.encode(v, add_special_tokens=False))
    label_ids[label] = list(set(ids))

n_pos = len(label_ids["positive"])
n_neg = len(label_ids["negative"])
all_label_ids = label_ids["positive"] + label_ids["negative"] + label_ids["neutral"]

ASSET_NAME = {"TSLA": "Tesla (TSLA)", "BTC-USD": "Bitcoin (BTC)"}.get(ASSET, ASSET)
SYSTEM_PROMPT = (
    f"You are a financial sentiment analyst. "
    f"Classify the sentiment towards {ASSET_NAME}. "
    f"Reply with exactly one word: positive, negative, or neutral."
)

def score_headline(text: str) -> float:
    """Score one headline. Returns P(+) - P(-) ∈ [-1, 1]."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Headline: {text[:300]}\nCompany: {ASSET_NAME}\nSentiment:"}
    ]
    try:
        prompt = sent_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{SYSTEM_PROMPT}\n\nHeadline: {text[:300]}\nCompany: {ASSET_NAME}\nSentiment:"

    inputs = sent_tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(DEVICE)
    with torch.no_grad():
        logits = sent_model(**inputs).logits[0, -1, :]

    label_logits = logits[all_label_ids]
    probs = torch.softmax(label_logits, dim=-1).cpu()
    return float(probs[:n_pos].sum() - probs[n_pos:n_pos+n_neg].sum())

# For demo: generate random sentiment (replace with actual news data)
# In production: load your news CSV and call score_headline() per article
print("Generating sentiment scores (demo: random — replace with news data)...")
np.random.seed(42)

def make_sentiment_series(df: pd.DataFrame) -> pd.Series:
    """Create daily sentiment. Replace with actual article scoring."""
    # DEMO: random scores
    # PRODUCTION: load news CSVs, filter by asset, call score_headline()
    scores = np.random.uniform(-0.3, 0.3, len(df))
    return pd.Series(scores.tolist(), index=df["date"].tolist())

train_sent = make_sentiment_series(train_df)
val_sent   = make_sentiment_series(val_df)
test_sent  = make_sentiment_series(test_df)
print(f"✅ Sentiment scores ready")

del sent_model, sent_base   # free GPU memory
gc.collect(); torch.cuda.empty_cache()

# %% CELL 6 — Feature Engineering
def add_features(df: pd.DataFrame, sentiment: pd.Series,
                 warmup: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Build 15-dim state matrix and daily returns."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    c, v = df["close"], df["volume"]

    # Technical indicators
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi14 = 100 - (100 / (1 + gain / (loss + 1e-10)))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()

    df["log_ret"]   = np.log(c / c.shift(1) + 1e-10)
    df["daily_ret"] = c.pct_change()
    df["ema10_f"]   = ema10 / c - 1
    df["ema50_f"]   = ema50 / c - 1
    df["rsi_f"]     = rsi14 / 50 - 1
    df["macd_f"]    = (macd - sig) / (c + 1e-10)
    df["boll_f"]    = (c - sma20) / (2 * std20 + 1e-10)
    df["vol_f"]     = np.log(v / v.shift(1) + 1e-10)

    # Calendar
    dates = pd.DatetimeIndex(df["date"])
    df["dow_sin"] = np.sin(2*np.pi*dates.dayofweek/7)
    df["dow_cos"] = np.cos(2*np.pi*dates.dayofweek/7)
    df["dom_sin"] = np.sin(2*np.pi*dates.day/31)
    df["dom_cos"] = np.cos(2*np.pi*dates.day/31)
    df["moy_sin"] = np.sin(2*np.pi*dates.month/12)
    df["moy_cos"] = np.cos(2*np.pi*dates.month/12)

    # Sentiment
    df["sentiment"] = df["date"].map(dict(zip(sentiment.index, sentiment.values))).fillna(0.0)

    # Drop warmup + NaN
    df = df.iloc[warmup:].dropna().reset_index(drop=True)

    FEAT_COLS = ["log_ret","ema10_f","ema50_f","rsi_f","macd_f","boll_f","vol_f","sentiment",
                 "dow_sin","dow_cos","dom_sin","dom_cos","moy_sin","moy_cos"]

    states = np.zeros((len(df), 15), dtype=np.float32)
    states[:, 0]  = df["log_ret"]
    states[:, 1]  = 0.0            # position filled by env
    states[:, 2]  = df["dow_sin"]
    states[:, 3]  = df["dow_cos"]
    states[:, 4]  = df["dom_sin"]
    states[:, 5]  = df["dom_cos"]
    states[:, 6]  = df["moy_sin"]
    states[:, 7]  = df["moy_cos"]
    states[:, 8]  = df["ema10_f"]
    states[:, 9]  = df["ema50_f"]
    states[:, 10] = df["rsi_f"]
    states[:, 11] = df["macd_f"]
    states[:, 12] = df["boll_f"]
    states[:, 13] = df["vol_f"]
    states[:, 14] = df["sentiment"]
    states = np.clip(states, -10, 10)

    returns = df["daily_ret"].values.astype(np.float32)
    return states, returns

# Build with warmup prepended (matching paper)
val_full  = pd.concat([train_df.tail(WARMUP_DAYS), val_df]).reset_index(drop=True)
test_full = pd.concat([val_df.tail(WARMUP_DAYS), test_df]).reset_index(drop=True)

all_sent = pd.concat([train_sent, val_sent, test_sent])

train_states, train_rets = add_features(train_df,   train_sent)
val_states,   val_rets   = add_features(val_full,   all_sent)
test_states,  test_rets  = add_features(test_full,  all_sent)

print(f"✅ States: Train={len(train_states)} Val={len(val_states)} Test={len(test_states)}")

# %% CELL 7 — Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, states, returns, random_start=True, apply_cost=True):
        super().__init__()
        self.states    = states.astype(np.float32)
        self.returns   = returns.astype(np.float32)
        self.T         = len(states)
        self.rand      = random_start
        self.cost      = 0.002 if apply_cost else 0.0
        self.action_space      = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (15,), np.float32)
        self.pos_map   = {0: -1, 1: 0, 2: 1}
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t    = int(np.random.randint(1, max(2, self.T-100))) if self.rand else 1
        self.val  = INITIAL_VALUE
        self.prev = 0
        self.vals = [self.val]
        self.acts = [0]
        return self._obs(), {}

    def _obs(self):
        s = self.states[self.t].copy(); s[1] = float(self.prev); return s

    def step(self, action):
        pos = self.pos_map[action]
        ret = float(self.returns[self.t])
        c   = self.cost if pos != self.prev else 0.0
        new_val = max(self.val * (1 + pos*ret) * (1-c), 1.0)
        alpha_r = (np.log(new_val/self.val+1e-10) - np.log(1+ret+1e-10))
        self.val = new_val; self.prev = pos; self.t += 1
        self.vals.append(new_val); self.acts.append(pos)
        done = self.t >= self.T-1
        return (self._obs() if not done else np.zeros(15,np.float32)), alpha_r, done, False, {}

    def metrics(self):
        v   = np.array(self.vals)
        r   = np.diff(v)/v[:-1]
        sr  = float(np.mean(r)/(np.std(r)+1e-10)*np.sqrt(252))
        cr  = float((v[-1]/v[0]-1)*100)
        pk  = np.maximum.accumulate(v)
        mdd = float(abs(((v-pk)/pk).min())*100)
        return {"CR%": round(cr,2), "SR": round(sr,3), "MDD%": round(mdd,2)}

def make_env(s, r, random_start=True, cost=True):
    return TradingEnv(s, r, random_start, cost)

# %% CELL 8 — Networks
class LSTMPolicy(nn.Module):
    def __init__(self, h=128, nl=2, na=3):
        super().__init__()
        self.lstm = nn.LSTM(15, h, nl, batch_first=True, dropout=0.1 if nl>1 else 0)
        self.fc   = nn.Linear(h, na)
        self.h    = None; self.hd = h; self.nl = nl
    def init_h(self, dev):
        self.h = (torch.zeros(self.nl,1,self.hd).to(dev),torch.zeros(self.nl,1,self.hd).to(dev))
    def forward(self, x, rst=False):
        if rst or self.h is None: self.init_h(x.device)
        if x.dim()==1: x=x.unsqueeze(0).unsqueeze(0)
        out,self.h = self.lstm(x,self.h); self.h=tuple(h.detach() for h in self.h)
        return self.fc(out[:,−1,:]).squeeze(0)

class LSTMActorCritic(nn.Module):
    def __init__(self, h=128, nl=2, na=3):
        super().__init__()
        self.lstm = nn.LSTM(15, h, nl, batch_first=True, dropout=0.1 if nl>1 else 0)
        self.act  = nn.Linear(h, na); self.crt = nn.Linear(h, 1)
        self.h    = None; self.hd = h; self.nl = nl
    def init_h(self, dev):
        self.h = (torch.zeros(self.nl,1,self.hd).to(dev),torch.zeros(self.nl,1,self.hd).to(dev))
    def forward(self, x, rst=False):
        if rst or self.h is None: self.init_h(x.device)
        if x.dim()==1: x=x.unsqueeze(0).unsqueeze(0)
        out,self.h = self.lstm(x,self.h); self.h=tuple(h.detach() for h in self.h)
        h=out[:,−1,:].squeeze(0); return self.act(h), self.crt(h).squeeze()
    def seq(self, x):  # [T,15] → [T,na], [T,1]
        x=x.unsqueeze(0); self.init_h(x.device)
        out,self.h = self.lstm(x,self.h)
        return self.act(out.squeeze(0)), self.crt(out.squeeze(0))

def ffnn(inp, h, out, nl=2):
    lys=[nn.Linear(inp,h),nn.LayerNorm(h),nn.ReLU()]
    for _ in range(nl-1): lys+=[nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU()]
    lys.append(nn.Linear(h,out)); return nn.Sequential(*lys)

class QNet(nn.Module):
    def __init__(self, h=256, nl=2): super().__init__(); self.n=ffnn(15,h,3,nl)
    def forward(self,x): return self.n(x)

class Actor(nn.Module):
    def __init__(self, h=256, nl=2):
        super().__init__()
        lys=[nn.Linear(15,h),nn.LayerNorm(h),nn.ReLU()]
        for _ in range(nl-1): lys+=[nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU()]
        lys+=[nn.Linear(h,1),nn.Tanh()]; self.n=nn.Sequential(*lys)
    def forward(self,x): return self.n(x)
    def act(self,x):
        with torch.no_grad(): v=self.n(x.unsqueeze(0)).item()
        return 0 if v<-1/3 else (2 if v>1/3 else 1), v

class Critic(nn.Module):
    def __init__(self, h=256, nl=2): super().__init__(); self.n=ffnn(16,h,1,nl)
    def forward(self,s,a): return self.n(torch.cat([s,a],dim=-1))

class ReplayBuffer:
    def __init__(self, cap=10000):
        self.cap=cap; self.ptr=0; self.sz=0
        self.s=np.zeros((cap,15),np.float32); self.a=np.zeros(cap,np.int64)
        self.r=np.zeros(cap,np.float32); self.ns=np.zeros((cap,15),np.float32)
        self.d=np.zeros(cap,np.float32); self.ca=np.zeros(cap,np.float32)
    def push(self,s,a,r,ns,d,ca=0):
        self.s[self.ptr]=s; self.a[self.ptr]=a; self.r[self.ptr]=r
        self.ns[self.ptr]=ns; self.d[self.ptr]=d; self.ca[self.ptr]=ca
        self.ptr=(self.ptr+1)%self.cap; self.sz=min(self.sz+1,self.cap)
    def sample(self,bs,dev):
        i=np.random.choice(self.sz,bs,replace=False)
        return (torch.FloatTensor(self.s[i]).to(dev), torch.LongTensor(self.a[i]).to(dev),
                torch.FloatTensor(self.r[i]).to(dev), torch.FloatTensor(self.ns[i]).to(dev),
                torch.FloatTensor(self.d[i]).to(dev), torch.FloatTensor(self.ca[i]).to(dev))
    def __len__(self): return self.sz

# %% CELL 9 — Training Utilities
def eval_agent(agent, val_states, val_rets):
    env = make_env(val_states, val_rets, random_start=False, cost=False)
    obs,_ = env.reset()
    if hasattr(agent,'policy'): agent.policy.h=None
    if hasattr(agent,'ac'):     agent.ac.h=None
    done=False
    while not done:
        with torch.no_grad():
            if hasattr(agent,'policy'):
                lg=agent.policy(torch.FloatTensor(obs).to(DEVICE)); act=lg.argmax().item()
            elif hasattr(agent,'ac'):
                lg,_=agent.ac(torch.FloatTensor(obs).to(DEVICE)); act=lg.argmax().item()
            elif hasattr(agent,'online'):
                q=agent.online(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)); act=q.argmax().item()
            elif hasattr(agent,'actor'):
                act,_=agent.actor.act(torch.FloatTensor(obs).to(DEVICE))
        obs,_,done,_,_ = env.step(act)
    m = env.metrics()
    v = np.array(env.vals)
    r = np.diff(v)/v[:-1]
    sr = float(np.mean(r)/(np.std(r)+1e-10)*np.sqrt(252))
    return sr, m

# %% CELL 10 — PG Agent
class PGAgent:
    def __init__(self, h=128, nl=2, lr=1e-3, gamma=0.99, ent=0.01):
        self.gamma=gamma; self.ent=ent
        self.policy=LSTMPolicy(h,nl).to(DEVICE)
        self.opt=torch.optim.Adam(self.policy.parameters(),lr=lr,weight_decay=1e-4)
        self.lp=[]; self.rw=[]; self.en=[]
    def act(self,s,rst=False):
        lg=self.policy(torch.FloatTensor(s).to(DEVICE),rst)
        d=torch.distributions.Categorical(logits=lg); a=d.sample()
        self.lp.append(d.log_prob(a)); self.en.append(d.entropy()); return a.item()
    def store(self,r): self.rw.append(r)
    def update(self):
        if not self.rw: return
        G=0; ret=[]
        for r in reversed(self.rw): G=r+self.gamma*G; ret.insert(0,G)
        ret=torch.FloatTensor(ret).to(DEVICE)
        ret=(ret-ret.mean())/(ret.std()+1e-8)
        lp=torch.stack(self.lp); en=torch.stack(self.en)
        loss=-(lp*ret).mean()-self.ent*en.mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),1.0)
        self.opt.step(); self.lp=[]; self.rw=[]; self.en=[]

def train_pg(max_ep=MAX_EPISODES, h=128, nl=2, lr=1e-3, gamma=0.99, ent=0.01):
    agent=PGAgent(h,nl,lr,gamma,ent)
    best_sr=-np.inf; best_st=None; no_imp=0
    for ep in range(1,max_ep+1):
        env=make_env(train_states,train_rets); obs,_=env.reset()
        agent.policy.h=None; agent.act(obs,rst=True); done=False; tot=0
        while not done:
            a=agent.act(obs); obs,r,done,_,_=env.step(a); agent.store(r); tot+=r
        agent.update()
        if ep%EVAL_EVERY==0:
            sr,m=eval_agent(agent,val_states,val_rets)
            print(f"  [PG] ep={ep} | tot={tot:.3f} | val_SR={sr:.3f} | val_CR={m['CR%']:.1f}%")
            if sr>best_sr: best_sr=sr; best_st=copy.deepcopy(agent.policy.state_dict()); no_imp=0
            else:
                no_imp+=1
                if no_imp>=PATIENCE: print(f"  [PG] Early stop ep={ep}"); break
    if best_st: agent.policy.load_state_dict(best_st)
    print(f"  [PG] Best val SR: {best_sr:.3f}"); return agent, best_sr

# %% CELL 11 — PPO Agent
class PPOAgent:
    def __init__(self, h=128, nl=2, lr=3e-4, gamma=0.99, lam=0.95,
                 eps=0.2, vc=0.5, ec=0.01, epochs=4):
        self.gamma=gamma; self.lam=lam; self.eps=eps; self.vc=vc; self.ec=ec; self.ep=epochs
        self.ac=LSTMActorCritic(h,nl).to(DEVICE)
        self.opt=torch.optim.Adam(self.ac.parameters(),lr=lr,weight_decay=1e-4)
        self.s=[]; self.a=[]; self.lp=[]; self.r=[]; self.v=[]; self.d=[]
    def act(self,s,rst=False):
        st=torch.FloatTensor(s).to(DEVICE)
        with torch.no_grad(): lg,val=self.ac(st,rst)
        dist=torch.distributions.Categorical(logits=lg); a=dist.sample()
        self.s.append(s); self.a.append(a.item()); self.lp.append(dist.log_prob(a).item())
        self.v.append(val.item()); return a.item()
    def store(self,r,d): self.r.append(r); self.d.append(d)
    def update(self, lv=0.0):
        if len(self.r)<2: self.s=[]; self.a=[]; self.lp=[]; self.r=[]; self.v=[]; self.d=[]; return
        vals=self.v+[lv]; adv=[]; g=0
        for t in reversed(range(len(self.r))):
            d=(1-self.d[t]); delta=self.r[t]+self.gamma*vals[t+1]*d-vals[t]
            g=delta+self.gamma*self.lam*d*g; adv.insert(0,g)
        adv=torch.FloatTensor(adv).to(DEVICE); adv=(adv-adv.mean())/(adv.std()+1e-8)
        ret=adv+torch.FloatTensor(self.v).to(DEVICE)
        st=torch.FloatTensor(np.array(self.s)).to(DEVICE)
        at=torch.LongTensor(self.a).to(DEVICE)
        olp=torch.FloatTensor(self.lp).to(DEVICE)
        for _ in range(self.ep):
            self.ac.h=None; lg,nv=self.ac.seq(st)
            dist=torch.distributions.Categorical(logits=lg)
            ratio=(dist.log_prob(at)-olp).exp()
            al=-torch.min(ratio*adv,torch.clamp(ratio,1-self.eps,1+self.eps)*adv).mean()
            cl=F.mse_loss(nv.squeeze(),ret); ent=dist.entropy().mean()
            loss=al+self.vc*cl-self.ec*ent
            self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.ac.parameters(),1.0); self.opt.step()
        self.s=[]; self.a=[]; self.lp=[]; self.r=[]; self.v=[]; self.d=[]

def train_ppo(max_ep=MAX_EPISODES, h=128, nl=2, lr=3e-4, gamma=0.99, lam=0.95, eps=0.2):
    agent=PPOAgent(h,nl,lr,gamma,lam,eps)
    best_sr=-np.inf; best_st=None; no_imp=0
    for ep in range(1,max_ep+1):
        env=make_env(train_states,train_rets); obs,_=env.reset()
        agent.ac.h=None; done=False; tot=0; lv=0
        while not done:
            a=agent.act(obs); obs,r,done,_,_=env.step(a); agent.store(r,done); tot+=r
        if not done:
            with torch.no_grad(): _,lv=agent.ac(torch.FloatTensor(obs).to(DEVICE)); lv=lv.item()
        agent.update(lv)
        if ep%EVAL_EVERY==0:
            sr,m=eval_agent(agent,val_states,val_rets)
            print(f"  [PPO] ep={ep} | tot={tot:.3f} | val_SR={sr:.3f} | val_CR={m['CR%']:.1f}%")
            if sr>best_sr: best_sr=sr; best_st=copy.deepcopy(agent.ac.state_dict()); no_imp=0
            else:
                no_imp+=1
                if no_imp>=PATIENCE: print(f"  [PPO] Early stop ep={ep}"); break
    if best_st: agent.ac.load_state_dict(best_st)
    print(f"  [PPO] Best val SR: {best_sr:.3f}"); return agent, best_sr

# %% CELL 12 — DQL Agent
class DQLAgent:
    def __init__(self, h=256, nl=2, lr=1e-3, gamma=0.99,
                 buf=10000, bs=64, tgt_upd=100,
                 eps_s=1.0, eps_e=0.05, eps_d=5000):
        self.gamma=gamma; self.bs=bs; self.tgt_upd=tgt_upd
        self.eps=eps_s; self.eps_e=eps_e; self.eps_d=eps_d; self.eps_s=eps_s; self.steps=0
        self.online=QNet(h,nl).to(DEVICE); self.target=QNet(h,nl).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt=torch.optim.Adam(self.online.parameters(),lr=lr,weight_decay=1e-4)
        self.buf=ReplayBuffer(buf)
    def act(self,s):
        self.eps=max(self.eps_e,self.eps_s-self.steps/self.eps_d*(self.eps_s-self.eps_e))
        if np.random.rand()<self.eps: return np.random.randint(3)
        with torch.no_grad(): return self.online(torch.FloatTensor(s).unsqueeze(0).to(DEVICE)).argmax().item()
    def store(self,s,a,r,ns,d): self.buf.push(s,a,r,ns,d); self.steps+=1
    def update(self):
        if len(self.buf)<self.bs: return
        s,a,r,ns,d,_=self.buf.sample(self.bs,DEVICE)
        with torch.no_grad():
            na=self.online(ns).argmax(1,keepdim=True)
            tq=self.target(ns).gather(1,na).squeeze()
            tgt=r+self.gamma*tq*(1-d)
        cq=self.online(s).gather(1,a.unsqueeze(1)).squeeze()
        loss=F.smooth_l1_loss(cq,tgt)
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),1.0); self.opt.step()
        if self.steps%self.tgt_upd==0: self.target.load_state_dict(self.online.state_dict())

def train_dql(max_ep=MAX_EPISODES, h=256, nl=2, lr=1e-3, gamma=0.99):
    agent=DQLAgent(h,nl,lr,gamma)
    best_sr=-np.inf; best_st=None; no_imp=0
    for ep in range(1,max_ep+1):
        env=make_env(train_states,train_rets); obs,_=env.reset(); done=False; tot=0
        while not done:
            a=agent.act(obs); no,r,done,_,_=env.step(a); agent.store(obs,a,r,no,done)
            agent.update(); obs=no; tot+=r
        if ep%EVAL_EVERY==0:
            sr,m=eval_agent(agent,val_states,val_rets)
            print(f"  [DQL] ep={ep} | eps={agent.eps:.3f} | val_SR={sr:.3f} | val_CR={m['CR%']:.1f}%")
            if sr>best_sr: best_sr=sr; best_st=copy.deepcopy(agent.online.state_dict()); no_imp=0
            else:
                no_imp+=1
                if no_imp>=PATIENCE: print(f"  [DQL] Early stop ep={ep}"); break
    if best_st: agent.online.load_state_dict(best_st)
    print(f"  [DQL] Best val SR: {best_sr:.3f}"); return agent, best_sr

# %% CELL 13 — DDPG Agent
class OUNoise:
    def __init__(self, mu=0.0, th=0.15, sig=0.2): self.mu=mu; self.th=th; self.sig=sig; self.reset()
    def reset(self): self.s=self.mu
    def __call__(self): self.s+=self.th*(self.mu-self.s)+self.sig*np.random.randn(); return self.s

class DDPGAgent:
    def __init__(self, h=256, nl=2, la=1e-4, lc=1e-3, gamma=0.99,
                 tau=0.005, buf=10000, bs=64, ou_t=0.15, ou_s=0.2):
        self.gamma=gamma; self.tau=tau; self.bs=bs; self.steps=0
        self.actor=Actor(h,nl).to(DEVICE); self.critic=Critic(h,nl).to(DEVICE)
        self.actor_t=Actor(h,nl).to(DEVICE); self.critic_t=Critic(h,nl).to(DEVICE)
        self.actor_t.load_state_dict(self.actor.state_dict()); self.critic_t.load_state_dict(self.critic.state_dict())
        self.actor_t.eval(); self.critic_t.eval()
        self.opt_a=torch.optim.Adam(self.actor.parameters(),lr=la,weight_decay=1e-4)
        self.opt_c=torch.optim.Adam(self.critic.parameters(),lr=lc,weight_decay=1e-4)
        self.buf=ReplayBuffer(buf); self.noise=OUNoise(theta=ou_t,sig=ou_s)
    def _soft(self,o,t):
        for p,pt in zip(o.parameters(),t.parameters()): pt.data.copy_(self.tau*p.data+(1-self.tau)*pt.data)
    def act(self,s,explore=True):
        st=torch.FloatTensor(s).to(DEVICE); disc,ca=self.actor.act(st)
        if explore: ca=float(np.clip(ca+self.noise(),-1,1))
        disc=0 if ca<-1/3 else (2 if ca>1/3 else 1)
        return disc, ca
    def store(self,s,a,r,ns,d,ca): self.buf.push(s,a,r,ns,d,ca); self.steps+=1
    def update(self):
        if len(self.buf)<self.bs: return
        s,a,r,ns,d,ca=self.buf.sample(self.bs,DEVICE)
        with torch.no_grad():
            na=self.actor_t(ns); tq=self.critic_t(ns,na).squeeze(); tgt=r+self.gamma*tq*(1-d)
        cq=self.critic(s,ca.unsqueeze(1)).squeeze()
        cl=F.mse_loss(cq,tgt)
        self.opt_c.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(),1.0); self.opt_c.step()
        pa=self.actor(s); al=-self.critic(s,pa).mean()
        self.opt_a.zero_grad(); al.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(),1.0); self.opt_a.step()
        self._soft(self.actor,self.actor_t); self._soft(self.critic,self.critic_t)

def train_ddpg(max_ep=MAX_EPISODES, h=256, nl=2, la=1e-4, lc=1e-3, gamma=0.99, tau=0.005):
    agent=DDPGAgent(h,nl,la,lc,gamma,tau)
    best_sr=-np.inf; best_a=None; best_c=None; no_imp=0
    for ep in range(1,max_ep+1):
        env=make_env(train_states,train_rets); obs,_=env.reset(); agent.noise.reset(); done=False; tot=0
        while not done:
            a,ca=agent.act(obs); no,r,done,_,_=env.step(a); agent.store(obs,a,r,no,done,ca)
            agent.update(); obs=no; tot+=r
        if ep%EVAL_EVERY==0:
            sr,m=eval_agent(agent,val_states,val_rets)
            print(f"  [DDPG] ep={ep} | tot={tot:.3f} | val_SR={sr:.3f} | val_CR={m['CR%']:.1f}%")
            if sr>best_sr: best_sr=sr; best_a=copy.deepcopy(agent.actor.state_dict()); best_c=copy.deepcopy(agent.critic.state_dict()); no_imp=0
            else:
                no_imp+=1
                if no_imp>=PATIENCE: print(f"  [DDPG] Early stop ep={ep}"); break
    if best_a: agent.actor.load_state_dict(best_a); agent.critic.load_state_dict(best_c)
    print(f"  [DDPG] Best val SR: {best_sr:.3f}"); return agent, best_sr

# %% CELL 14 — Train All Agents
print("="*60)
print(f"Training all agents on {ASSET}")
print("="*60)

print("\n[1/4] Policy Gradient...")
pg_agent, pg_sr = train_pg()

print("\n[2/4] PPO...")
ppo_agent, ppo_sr = train_ppo()

print("\n[3/4] DQL...")
dql_agent, dql_sr = train_dql()

print("\n[4/4] DDPG...")
ddpg_agent, ddpg_sr = train_ddpg()

print(f"\n{'='*60}")
print("Val Sharpe Summary:")
print(f"  PG  : {pg_sr:.3f}")
print(f"  PPO : {ppo_sr:.3f}")
print(f"  DQL : {dql_sr:.3f}")
print(f"  DDPG: {ddpg_sr:.3f}")
best_model = max([("PG",pg_sr),("PPO",ppo_sr),("DQL",dql_sr),("DDPG",ddpg_sr)],key=lambda x:x[1])
print(f"\n🏆 Best by val SR: {best_model[0]} (SR={best_model[1]:.3f})")

# %% CELL 15 — Final Evaluation (Test Set — touch once)
def full_eval(agent, name):
    env = make_env(test_states, test_rets, random_start=False, cost=False)
    obs,_ = env.reset()
    if hasattr(agent,'policy'): agent.policy.h=None
    if hasattr(agent,'ac'):     agent.ac.h=None
    done=False
    while not done:
        with torch.no_grad():
            if hasattr(agent,'policy'): a=agent.policy(torch.FloatTensor(obs).to(DEVICE)).argmax().item()
            elif hasattr(agent,'ac'):   lg,_=agent.ac(torch.FloatTensor(obs).to(DEVICE)); a=lg.argmax().item()
            elif hasattr(agent,'online'): a=agent.online(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)).argmax().item()
            else: a,_=agent.actor.act(torch.FloatTensor(obs).to(DEVICE))
        obs,_,done,_,_ = env.step(a)
    m = env.metrics()
    print(f"  {name:8s} | Test CR={m['CR%']:6.1f}% | SR={m['SR']:6.3f} | MDD={m['MDD%']:5.1f}%")
    return m

print("\n" + "="*60)
print(f"FINAL TEST SET RESULTS — {ASSET}")
print("="*60)

# Buy & Hold
bnh = make_env(test_states, test_rets, random_start=False, cost=False); bnh.reset()
for _ in range(bnh.T-1): bnh.step(2)
bm = bnh.metrics()
print(f"  {'B&H':8s} | Test CR={bm['CR%']:6.1f}% | SR={bm['SR']:6.3f} | MDD={bm['MDD%']:5.1f}%")

pg_m   = full_eval(pg_agent,   "PG")
ppo_m  = full_eval(ppo_agent,  "PPO")
dql_m  = full_eval(dql_agent,  "DQL")
ddpg_m = full_eval(ddpg_agent, "DDPG")

results = pd.DataFrame([
    {"Model":"B&H",  **bm},
    {"Model":"PG",   **pg_m},
    {"Model":"PPO",  **ppo_m},
    {"Model":"DQL",  **dql_m},
    {"Model":"DDPG", **ddpg_m},
])
results.to_csv(f"{SAVE_DIR}/results_{ASSET}.csv", index=False)
print(f"\n✅ Saved to {SAVE_DIR}/results_{ASSET}.csv")
print(results.to_string(index=False))
