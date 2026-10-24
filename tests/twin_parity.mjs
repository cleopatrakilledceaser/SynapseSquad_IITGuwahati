// Parity: the browser twin (web/index.html, TWIN-CORE block) against Python references built from
// src/mech/model.py (tests/fixtures/twin_parity.json, written by src/web/twin_live.py).
// Checks (1) EKF forecasts at many anchors (model.ekf_forecast), including a case with sensor gaps,
// and (2) the what-if path the page uses (meal now, assumed walk with the one-sided kernel), including
// the README's S-151 example. Run: node tests/twin_parity.mjs
import { readFileSync } from "node:fs";
const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
const a = html.indexOf("/* TWIN-CORE-BEGIN"), b = html.indexOf("/* TWIN-CORE-END */");
if (a < 0 || b < 0) throw new Error("TWIN-CORE markers not found");
const TWIN = new Function(html.slice(a, b) + "\nreturn TWIN;")();
const fx = JSON.parse(readFileSync(new URL("./fixtures/twin_parity.json", import.meta.url), "utf8"));
const M = TWIN.model(fx.meta), hs = [6, 12, 24];
let worstM = 0, worstV = 0, n = 0, worstS = 0, ns = 0, walks = 0, gaps = 0;
for (const c of fx.cases) {
  const A = TWIN.arrays(c);
  gaps += A.y.filter((v) => v !== v).length;
  if (c.anchors) {
    const out = TWIN.ekfForecast(c.theta, A.y, A.carbs, A.slow, A.act, c.anchors, hs, M);
    out.forEach((o, k) => o.mean.forEach((m, j) => {
      worstM = Math.max(worstM, Math.abs(m - c.mean[k][j]));
      worstV = Math.max(worstV, Math.abs(o.var[j] - c.var[k][j]) / Math.max(1, c.var[k][j]));
      n++;
    }));
  }
  if (c.scenarios) {
    const f = TWIN.filter(c.theta, A.y, A.carbs, A.slow, A.act, M), i = c.scenario_i;
    for (const sc of c.scenarios) {
      const r = TWIN.scenario({ ...f[i], slowHeld: A.slow[i] }, c.theta, M, sc, fx.meta.walk, 24);
      r.mean.forEach((m, j) => { worstS = Math.max(worstS, Math.abs(m - sc.mean[j])); ns++; });
      if (sc.walk) walks++;
    }
  }
}
console.log(`${n} EKF forecasts: max |mean diff| ${worstM.toExponential(2)} mg/dL, max rel var diff ${worstV.toExponential(2)}`);
console.log(`${ns} what-if steps (${walks} scenarios with a walk, ${gaps} missing readings): max |diff| ${worstS.toExponential(2)} mg/dL`);
const ex = fx.cases.find((c) => c.id === "S-151");
if (ex) console.log("S-151 peaks:", ex.scenarios.map((s) => `${s.label} ${s.peak.toFixed(1)}`).join(" | "));
if (!(worstM < 1e-6 && worstV < 1e-6 && worstS < 1e-6 && walks > 0 && gaps > 0)) { console.error("PARITY FAILED"); process.exit(1); }
console.log("PARITY OK");
