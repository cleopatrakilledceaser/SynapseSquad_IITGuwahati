// Parity: the browser twin (web/index.html, TWIN-CORE block) against the Python reference
// (src/mech/model.py ekf_forecast) on tests/fixtures/twin_parity.json. Run: node tests/twin_parity.mjs
import { readFileSync } from "node:fs";
const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
const a = html.indexOf("/* TWIN-CORE-BEGIN"), b = html.indexOf("/* TWIN-CORE-END */");
if (a < 0 || b < 0) throw new Error("TWIN-CORE markers not found");
const TWIN = new Function(html.slice(a, b) + "\nreturn TWIN;")();
const fx = JSON.parse(readFileSync(new URL("./fixtures/twin_parity.json", import.meta.url), "utf8"));
const M = TWIN.model(fx.meta), hs = [6, 12, 24];
let worstM = 0, worstV = 0, n = 0;
for (const c of fx.cases) {
  const A = TWIN.arrays(c);
  const out = TWIN.ekfForecast(c.theta, A.y, A.carbs, A.slow, A.act, c.anchors, hs, M);
  out.forEach((o, k) => o.mean.forEach((m, j) => {
    worstM = Math.max(worstM, Math.abs(m - c.mean[k][j]));
    worstV = Math.max(worstV, Math.abs(o.var[j] - c.var[k][j]) / Math.max(1, c.var[k][j]));
    n++;
  }));
}
console.log(`${fx.cases.length} patients, ${n} forecasts: max |mean diff| ${worstM.toExponential(2)} mg/dL, max rel var diff ${worstV.toExponential(2)}`);
if (!(worstM < 1e-6 && worstV < 1e-6)) { console.error("PARITY FAILED"); process.exit(1); }
console.log("PARITY OK");
