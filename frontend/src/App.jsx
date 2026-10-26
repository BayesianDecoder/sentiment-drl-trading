import StrategyView from "./components/StrategyView";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>A2C Trading Strategy</h1>
        <p className="header-sub">Sentiment-fused DRL agent · NIFTY-50 · live backtest results</p>
      </header>
      <main className="app-main">
        <StrategyView />
      </main>
    </div>
  );
}
