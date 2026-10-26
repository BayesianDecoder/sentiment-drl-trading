import { getStockMeta } from "../stockMeta";

export default function StockAvatar({ ticker, size = 36 }) {
  const meta = getStockMeta(ticker);
  return (
    <div
      className="stock-avatar"
      style={{
        width: size, height: size, background: meta.color,
        fontSize: size * 0.36, borderRadius: size * 0.28,
      }}
      title={meta.name}
    >
      {meta.initials}
    </div>
  );
}
