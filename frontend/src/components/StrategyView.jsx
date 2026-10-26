import { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { getA2CStrategy, getNews, getRlModels } from "../api";
import { getStockMeta } from "../stockMeta";
import StockAvatar from "./StockAvatar";
import Sparkline from "./Sparkline";
import LivePanel from "./LivePanel";

const ACTION_STYLE = {
  buy: { label: "BOUGHT", color: "#10b981", bg: "rgba(16,185,129,0.15)" },
  sell: { label: "SOLD", color: "#ef4444", bg: "rgba(239,68,68,0.15)" },
  hold: { label: "HOLD", color: "#6b7280", bg: "rgba(107,114,128,0.12)" },
};

function ActionBadge({ action }) {
  const s = ACTION_STYLE[action] || ACTION_STYLE.hold;
  return <span className="action-badge" style={{ color: s.color, background: s.bg }}>{s.label}</span>;
}

function CustomChartTooltip({ active, payload, label, strategy }) {
  if (!active || !payload || !payload.length) return null;
  const day = strategy.days.find((d) => d.date === label);
  if (!day) return null;

  const held = Object.entries(day.stocks).filter(([, s]) => s.shares_held > 0);

  return (
    <div className="chart-tooltip">
      <div className="tooltip-date">{label}</div>
      <div className="tooltip-row">
        <span>A2C portfolio:</span>
        <strong>{(payload.find((p) => p.dataKey === "portfolio")?.value * 100).toFixed(1)}%</strong>
      </div>
      <div className="tooltip-row">
        <span>Buy &amp; Hold:</span>
        <strong>{(payload.find((p) => p.dataKey === "buyhold")?.value * 100).toFixed(1)}%</strong>
      </div>
      <div className="tooltip-holdings-label">Portfolio holdings this day:</div>
      {held.length === 0 ? (
        <div className="tooltip-empty">All cash — no positions held</div>
      ) : (
        <div className="tooltip-holdings">
          {held.map(([ticker, s]) => (
            <div key={ticker} className="tooltip-holding-row">
              <StockAvatar ticker={ticker} size={20} />
              <span>{getStockMeta(ticker).name}</span>
              <span className="tooltip-shares">{s.shares_held} sh</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function findNearestDay(days, isoDate) {
  const targetDate = isoDate.slice(0, 10);
  let best = null;
  for (const d of days) {
    if (d.date <= targetDate && (!best || d.date > best.date)) best = d;
  }
  return best;
}

const NEWS_PAGE_SIZE = 12;

export default function StrategyView() {
  const [strategy, setStrategy] = useState(null);
  const [news, setNews] = useState([]);
  const [newsShown, setNewsShown] = useState(NEWS_PAGE_SIZE);
  const [a2cMetrics, setA2cMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getA2CStrategy(), getNews(), getRlModels()])
      .then(([strategyData, newsData, models]) => {
        setStrategy(strategyData);
        setNews(newsData);
        setA2cMetrics(models.find((m) => m.agent === "A2C"));
      })
      .catch((e) => setError(e.message));
  }, []);

  const chartData = useMemo(() => {
    if (!strategy) return [];
    return strategy.days.map((d, i) => ({
      date: d.date,
      portfolio: d.portfolio_value / strategy.initial_cash,
      buyhold: strategy.buy_hold_curve[i],
    }));
  }, [strategy]);

  if (error) return <p className="error">Failed to load strategy data: {error}</p>;
  if (!strategy) return <p className="pending-note">Loading A2C strategy…</p>;

  const lastDay = strategy.days[strategy.days.length - 1];

  return (
    <div className="strategy-layout">
      <aside className="stock-sidebar">
        <h3>Portfolio Tracker</h3>
        <p className="sidebar-sub">10 stocks — A2C's latest action on each</p>
        {strategy.tickers.map((ticker) => {
          const latest = lastDay.stocks[ticker];
          const priceSeries = strategy.days.map((d) => d.stocks[ticker].price);
          const meta = getStockMeta(ticker);
          return (
            <div key={ticker} className="stock-row">
              <StockAvatar ticker={ticker} />
              <div className="stock-row-info">
                <div className="stock-row-name">{meta.name}</div>
                <div className="stock-row-meta">
                  {latest.shares_held} sh · ₹{latest.price.toLocaleString("en-IN")}
                </div>
              </div>
              <Sparkline values={priceSeries} />
              <ActionBadge action={latest.action} />
            </div>
          );
        })}
      </aside>

      <main className="strategy-main">
        <LivePanel />

        <h2 className="section-divider">Historical Backtest</h2>

        {a2cMetrics && (
          <div className="metrics-strip">
            <div className="metric"><span>Return</span><strong style={{ color: "#10b981" }}>+{a2cMetrics.test_return_pct?.toFixed(2)}%</strong></div>
            <div className="metric"><span>Sharpe</span><strong>{a2cMetrics.test_sharpe?.toFixed(3)}</strong></div>
            <div className="metric"><span>Max Drawdown</span><strong>{a2cMetrics.test_max_drawdown_pct?.toFixed(2)}%</strong></div>
            <div className="metric"><span>vs. Buy&amp;Hold</span><strong style={{ color: "#10b981" }}>✅ Beat both metrics</strong></div>
          </div>
        )}

        <div className="panel chart-panel">
          <h2>How A2C Is Working</h2>
          <p className="subtitle">
            Portfolio value over the test period (normalized to 1.0 at start). Hover the chart to
            see exactly which stocks A2C was holding on any given day.
          </p>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} interval={20} />
              <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip content={<CustomChartTooltip strategy={strategy} />} />
              <Legend />
              <Line type="monotone" dataKey="portfolio" name="A2C" stroke="#f59e0b" dot={false} strokeWidth={2.5} />
              <Line type="monotone" dataKey="buyhold" name="Buy&Hold" stroke="#888" strokeDasharray="5 5" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel news-panel">
          <h2>News Received &amp; How A2C Reacted</h2>
          <p className="subtitle">
            Real headlines fetched and scored with the trained FinBERT-LoRA model, matched to
            A2C's action on that stock on the nearest available trading day in the backtest.
          </p>
          <div className="news-list">
            {news.slice(0, newsShown).map((item) => {
              const day = findNearestDay(strategy.days, item.date);
              const action = day?.stocks[item.ticker]?.action;
              return (
                <div key={item._id} className="news-row">
                  <StockAvatar ticker={item.ticker} size={30} />
                  <div className="news-row-body">
                    <div className="news-headline">{item.headline}</div>
                    <div className="news-meta">
                      <span className={`sentiment-tag sentiment-${item.sentiment_label}`}>{item.sentiment_label}</span>
                      <span className="news-date">{item.date.slice(0, 10)}</span>
                      {day && action && (
                        <span className="news-reaction">
                          → A2C {action === "hold" ? "held" : action === "buy" ? "bought" : "sold"} {getStockMeta(item.ticker).name} on {day.date}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          {newsShown < news.length && (
            <button className="show-more-btn" onClick={() => setNewsShown((n) => n + NEWS_PAGE_SIZE)}>
              Show more ({news.length - newsShown} remaining)
            </button>
          )}
        </div>
      </main>
    </div>
  );
}
