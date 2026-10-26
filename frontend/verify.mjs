import { chromium } from "playwright";

const app = await chromium.launch({ args: ["--no-sandbox"] });
const page = await app.newPage({ viewport: { width: 1400, height: 1000 } });
const consoleErrors = [];
page.on("console", (msg) => { if (msg.type() === "error") consoleErrors.push(msg.text()); });
page.on("pageerror", (err) => consoleErrors.push("pageerror: " + err.message));

await page.goto("http://localhost:5173", { waitUntil: "networkidle" });
await page.waitForSelector("text=A2C Trading Strategy");
await page.waitForSelector("text=Portfolio Tracker");
await page.waitForTimeout(1000);
await page.screenshot({ path: "/tmp/strategy_full.png", fullPage: true });

// Hover the chart to trigger the custom tooltip
const chart = await page.$(".recharts-surface");
if (chart) {
  const box = await chart.boundingBox();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.4);
  await page.waitForTimeout(500);
  await page.screenshot({ path: "/tmp/strategy_tooltip.png" });
}

console.log("CONSOLE_ERRORS:", JSON.stringify(consoleErrors));
await app.close();
