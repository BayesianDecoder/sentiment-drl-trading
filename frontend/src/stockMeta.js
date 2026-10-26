// Display metadata for the 10-stock universe. No reliable logo source was available to embed
// safely (external logo APIs can 404 or serve the wrong mark), so each stock gets a clean,
// self-contained colored badge with its initials instead of a fetched image.
export const STOCK_META = {
  "INFY.NS": { name: "Infosys", sector: "IT", initials: "IN", color: "#3B82F6" },
  "RELIANCE.NS": { name: "Reliance", sector: "Energy", initials: "RI", color: "#F97316" },
  "TCS.NS": { name: "TCS", sector: "IT", initials: "TC", color: "#2563EB" },
  "SBIN.NS": { name: "SBI", sector: "Banking", initials: "SB", color: "#6366F1" },
  "MARUTI.NS": { name: "Maruti", sector: "Auto", initials: "MS", color: "#EF4444" },
  "WIPRO.NS": { name: "Wipro", sector: "IT", initials: "WI", color: "#8B5CF6" },
  "TMPV.NS": { name: "Tata Motors", sector: "Auto", initials: "TM", color: "#0EA5E9" },
  "LT.NS": { name: "L&T", sector: "Infra", initials: "LT", color: "#14B8A6" },
  "ITC.NS": { name: "ITC", sector: "FMCG", initials: "IT", color: "#A855F7" },
  "SUNPHARMA.NS": { name: "Sun Pharma", sector: "Pharma", initials: "SP", color: "#F59E0B" },
};

export const getStockMeta = (ticker) =>
  STOCK_META[ticker] || { name: ticker, sector: "", initials: ticker.slice(0, 2).toUpperCase(), color: "#6b7280" };
