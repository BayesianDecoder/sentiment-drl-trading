import { useEffect, useState } from "react";
import { getLiveStatus, getLiveState } from "../api";
import { getStockMeta } from "../stockMeta";
import StockAvatar from "./StockAvatar";

const REFRESH_MS = 15_000;   // how often the DASHBOARD checks for updates -- the backend itself
                              // only actually re-polls yfinance/FinBERT every 90s; checking more
                              // often here just catches that update sooner, it doesn't cause extra
                              // external API calls.

const ACTION_STYLE = {
  buy: { label: "BUY", color: "#10b981", bg: "rgba(16,185,129,0.15)" },
  sell: { label: "SELL", color: "#ef4444", bg: "rgba(239,68,68,0.15)" },
  hold: { label: "HOLD", color: "#6b7280", bg: "rgba(107,114,128,0.12)" },
};

function timeAgo(iso) {
  if (!iso) return "never";
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const mins = Math.floor(seconds / 60);
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ago`;
}

export default function LivePanel() {
  const [status, setStatus] = useState(null);
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [, forceTick] = useState(0);   // re-render every second so "Xs ago" stays live

  useEffect(() => {
    const fetchAll = () => {
      Promise.all([getLiveStatus(), getLiveState()])
        .then(([s, st]) => { setStatus(s); setState(st); setError(null); })
        .catch((e) => setError(e.message));
    };
    fetchAll();
    const dataInterval = setInterval(fetchAll, REFRESH_MS);
    const tickInterval = setInterval(() => forceTick((n) => n + 1), 1000);
    return () => { clearInterval(dataInterval); clearInterval(tickInterval); };
  }, []);

  if (error) {
    return (
      <div className="panel live-panel">
        <p className="error">Live pipeline unreachable: {error} — is the FastAPI service running on :8000?</p>
      </div>
    );
  }
  if (!status || !state) {
    return <div className="panel live-panel"><p className="pending-note">Connecting to live pipeline…</p></div>;
  }

  const tickers = Object.keys(state.prices || {});

  return (
    <div className="panel live-panel">
      <div className="live-header">
        <span className={`live-dot ${status.market_open ? "live-dot-on" : "live-dot-off"}`} />
        <h2>Live</h2>
        <span className="live-market-badge">{status.market_open ? "NSE Open" : "NSE Closed"}</span>
        <span className="live-updated">Updated {timeAgo(status.last_updated)} · polls every {status.poll_interval_seconds}s</span>
      </div>
      <p className="subtitle">
        Real intraday prices and real news, fetched and scored just now — A2C's recommendation
        column is a fresh forward pass on today's live data, not the historical backtest replayed.
      </p>

      {tickers.length > 0 ? (
        <table className="data-table live-table">
          <thead>
            <tr><th>Stock</th><th>Live Price</th><th>Live Sentiment</th><th>A2C Says</th></tr>
          </thead>
          <tbody>
            {tickers.map((ticker) => {
              const sentiment = state.sentiment[ticker];
              const action = state.recommended_actions[ticker];
              const style = ACTION_STYLE[action] || ACTION_STYLE.hold;
              return (
                <tr key={ticker}>
                  <td><div className="live-stock-cell"><StockAvatar ticker={ticker} size={26} />{getStockMeta(ticker).name}</div></td>
                  <td>₹{state.prices[ticker]?.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</td>
                  <td style={{ color: sentiment > 0.1 ? "#10b981" : sentiment < -0.1 ? "#ef4444" : "#9ca3af" }}>
                    {sentiment !== undefined ? sentiment.toFixed(3) : "—"}
                  </td>
                  <td><span className="action-badge" style={{ color: style.color, background: style.bg }}>{style.label}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <p className="pending-note">Waiting for the first live poll cycle to complete…</p>
      )}

      {state.news && state.news.length > 0 && (
        <div className="live-news">
          <h3>Just in</h3>
          {state.news.slice(0, 5).map((n, i) => (
            <div key={i} className="news-row">
              <StockAvatar ticker={n.ticker} size={26} />
              <div className="news-row-body">
                <div className="news-headline">{n.headline}</div>
                <div className="news-meta">
                  <span className={`sentiment-tag sentiment-${n.sentiment_label}`}>{n.sentiment_label}</span>
                  <span className="news-date">fetched {timeAgo(n.fetched_at)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
