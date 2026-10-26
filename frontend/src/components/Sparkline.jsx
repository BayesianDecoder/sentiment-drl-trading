export default function Sparkline({ values, width = 90, height = 28, color = "#6366f1" }) {
  if (!values || values.length < 2) return <svg width={width} height={height} />;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = width / (values.length - 1);

  const points = values
    .map((v, i) => `${(i * stepX).toFixed(1)},${(height - ((v - min) / range) * height).toFixed(1)}`)
    .join(" ");

  const trendUp = values[values.length - 1] >= values[0];

  return (
    <svg width={width} height={height}>
      <polyline points={points} fill="none" stroke={trendUp ? "#10b981" : "#ef4444"} strokeWidth="1.5" />
    </svg>
  );
}
