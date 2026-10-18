# GlucoTwin clinician dashboard: design plan (Part 9, checkpoint 1)

Status: PLAN ONLY. No UI code exists yet. Waiting for owner approval before Step 3.
Explainability (Step 1) is built: `src/explain`, `configs/explain.yaml`, D-057. Its unit tests are
written in Step 5, after approval.

## 1. Audience and job

**Audience.** A diabetologist or physician with about two minutes per patient, in a bright
consulting room, on a 1440 px laptop or a tablet held in landscape (820 px wide). They know
glucose targets (70, 140, 180 mg/dL). They do not know what SHAP, conformal or a twin is.

**Primary job.** "Is this patient about to go high or low, how sure is the model, and what happens
if the patient changes a meal or walks?"

Everything on screen answers one of three questions, in this order:
1. What is likely to happen in the next 2 hours? (the forecast chart plus one plain sentence)
2. How far can I trust that? (interval width, data quality, confidence label)
3. What would change it? (what-if lab, with its limits stated next to the result)

Pages that are not about those three questions (twin state, personalization, input tiers, data
quality, evidence, export) are reference pages. They are quiet, text-led and reached from the
left navigation. They never compete with the chart.

## 2. Design tokens

### Colour

Six interface colours, all named. The brand colour is a deep plum ("damson"). It is used for
interaction (links, focus ring, selected state, primary button) and for the forecast: the fan is
the model's voice, so it wears the brand colour. It is not used for any glucose state.

| token | hex | role | contrast on white |
|---|---|---|---|
| paper | `#F5F6F8` | page background | n/a |
| surface | `#FFFFFF` | the chart frame and inputs only | n/a |
| ink | `#1B2230` | body text, history line | 15.9 : 1 |
| slate | `#566174` | secondary text, axis labels | 6.3 : 1 |
| rule | `#D8DDE5` | hairlines, borders | n/a (decorative) |
| damson | `#43285E` | brand, interaction, forecast | 12.3 : 1 |

Semantic glucose colours. Fixed, never reused for anything else, never used as a text colour (the
amber and mint fail text contrast on white, so text beside them is always ink):

| state | range | hex | L* | pattern (colour is never the only signal) | label |
|---|---|---|---|---|---|
| low | below 70 mg/dL | `#2B6CB0` | 44.8 | diagonal hatch | "Low" |
| in range | 70 to 180 mg/dL | `#9AD3B8` | 80.1 | plain tint | "In range" |
| high | above 180 mg/dL | `#E09A12` | 68.8 | dots | "High" |
| very high | above 250 mg/dL | `#8C1D2B` | 30.9 | cross-hatch | "Very high" |

Measured (Machado 2009 simulation, severity 1.0, script `logs/p9_colors.py`):
- Smallest CIELAB distance between any two of the four states: 66.7 for normal vision, 48.9
  deuteranopia, 55.1 protanopia, 34.5 tritanopia.
- Smallest lightness gap between any two states: 8.0 L* (in range vs high, deuteranopia). That gap
  is small, which is why every state also has a pattern and a text label.
- Damson vs the four glucose colours: smallest distance 27.5 across all four vision types (damson
  vs low, deuteranopia). The brand colour is distinguishable but not far, so the forecast fan is
  drawn as a band with a solid median line and is never confused with a zone fill.

The 250 mg/dL "very high" threshold is a DISPLAY band only (models and alerts use 70, 140, 180 per
D-020). It is labelled "above 250 mg/dL" wherever shown and no risk is computed for it.

### Typefaces

- **Atkinson Hyperlegible** (Braille Institute, SIL OFL), weights 400 and 700. All UI text and
  every number. It was designed for letter and digit distinguishability (1/l/I, 0/O, 6/b), which
  matters when a value is read at a glance on a washed-out screen.
- **Source Serif 4** (Adobe, SIL OFL), semibold 600 only. Page titles and the one-sentence decision
  strip. The serif gives the decision a printed clinical-note voice and separates "what the system
  says" from "labels and controls" without extra colour.
- Both are bundled as woff2 in `app/static/fonts/` (already added, licences in `NOTICE.md`),
  loaded with `@font-face` from Streamlit static serving. Fallbacks: `"Segoe UI", system-ui,
  sans-serif` and `Georgia, serif`. No CDN, no Google Fonts. Plotly uses the same families.
- Not yet verified: whether Atkinson's digits are fixed-width. If not, numbers that change on
  every replay tick sit in a right-aligned box of fixed minimum width so the layout does not jump.

### Type scale (px at 1440; at tablet width the two largest steps scale by 0.85)

| step | size / line height | use |
|---|---|---|
| caption | 12 / 16, slate | units, axis labels, footnotes |
| body-s | 14 / 20 | table text, helper text |
| body | 16 / 24 | default |
| lead | 20 / 28, serif 600 | section headings |
| title | 28 / 34, serif 600 | page titles, decision sentence |
| figure | 40 / 44, sans 700 | the one large readout (event probability) |

Sentence case everywhere. No all-caps and no letter-spacing tricks.

### Spacing and radius

- Spacing scale (px): 4, 8, 12, 16, 24, 32, 48, 64. Page gutter 32 (tablet 20). Space between
  sections 48, inside a section 16 to 24.
- Radius scale (px): 0 (rules, tables, bands), 3 (inputs, buttons, badges), 8 (the chart frame and
  nothing else). Only the chart frame has a surface fill and border. All other content sits on the
  paper and is separated by hairlines and whitespace, not boxes.
- Focus: 2 px damson outline with 2 px offset on every interactive element. Minimum target 40 px.

### Motion

One deliberate moment: when a patient opens, the forecast fan draws in from "now" to +120 min
(about 900 ms, ease-out, a reveal on the fan traces only). Motion that answers a click is allowed:
the what-if ghost trajectory fades in over 200 ms after "Run what-if". Everything else is static,
including hover. `prefers-reduced-motion: reduce` turns both off. Replay updates the chart in
place and never replays the draw-in.

### Dark mode

Not included. The semantic palette and pattern fills are validated on a light background only, and
clinicians work in bright rooms. A dark option would need a second validated colour-blind-safe set,
which is not cheap.


## 3. Layout

Shell (all pages): a slim 200 px left navigation (text links, the selected one marked by a 3 px
damson bar and bold weight), a content column up to 1200 px, and a footer line with the disclaimer
"Research prototype for decision support. Not a medical device. Not for diagnosis or treatment."
The patient selector and the synthetic label live in a top line above the page title. At 820 px the
navigation collapses to a horizontal row of links above the title; content becomes one column.

### a. Patient overview

One sentence: a patient list with one status line each on the left, and for the selected patient
the decision strip, a small static profile and the data-quality banner on the right.

```
+-------+--------------------------------------------------------------+
| nav   | Patients                                                     |
| Over- | P-014  T2D        Rising, 62% above 180 mg/dL by 11:40       |
| view  | P-021  Prediabet. Steady, low risk                           |
| Live  | S-001  SYNTHETIC  Falling, 18% below 70 mg/dL by 11:20       |
| What  |--------------------------------------------------------------|
| Twin  | Next 2 hours: glucose rising. 62% chance of going above      |
| Pers. | 180 mg/dL by 11:40. Warning time about 25 min.  [Moderate]   |
| Tiers | Why this alert  (drawer, closed)                             |
| Qual. |--------------------------------------------------------------|
| Evid. | Profile: age, sex, BMI, HbA1c, fasting glucose | Data quality|
| Export| banner: "Sensor data complete. Bias and drift not detectable"|
| ----- | Footer disclaimer                                            |
+-------+--------------------------------------------------------------+
```

### b. Live twin

One sentence: the hero forecast chart takes the full content width and about 60 percent of the
viewport height, with play, pause and speed above it, and three small event readouts and the alert
state below it.

```
| Live twin   P-014                      [Play] [Pause]  Speed [1x v] |
| +----------------------------------------------------------------+ |
| |  mg/dL                                  ..'''' fan 80% bands   | |
| |  250 ~~ very high (cross-hatch) ~~~~~~~~/~~~~~~~~~~~~~~~~~~~~~~ | |
| |  180 -- high (dots) ------------------/------------------------ | |
| |  history ________/\\____/\\__  NOW |  +30    +60     +120 min    | |
| |  70  -- target zone (tint) ---------------------------------- | |
| |  meal markers (triangles on the axis)                          | |
| +----------------------------------------------------------------+ |
| Above 180 mg/dL: 62%   Above 140: 81%   Below 70: 3%   (+60 min)    |
| Alert: ON, warning time about 25 min  | Confidence: moderate        |
| (when the interval is wide or data quality is low, a banner         |
|  replaces the alert: "Uncertain. Do not rely on this forecast.")    |
```

### c. What-if lab

One sentence: controls on the left (meal preset or custom grams, walk toggle, "Run what-if"), the
hero chart on the right with the ghost trajectory over the baseline, and under it a two-column
comparison, the fixed note, the twin size, and the plausibility pass rates.

```
| Controls         | +----------------------------------------------+ |
| Meal  [Roti v]   | | baseline (solid)   what-if (ghost, dashed)    | |
| Carbs 45 g       | +----------------------------------------------+ |
| Walk  [ ] 30 min | Baseline +60: 168   What-if +60: 151  (mg/dL)     |
| [Run what-if]    | Twin size: -12 mg/dL (twin, labelled)             |
|                  | Note: Model-based estimate, not a causal          |
|                  | guarantee. In matched real breakfasts ...         |
|                  | Plausibility pass rates (table)                   |
|                  | What this tool cannot tell you (panel, closed)    |
```

### d to i. Reference pages

One sentence each, shared skeleton: title, one plain paragraph, then one figure or table, then
caveats. No cards.
- **d. Twin state:** a single large "Basal glucose Gb" value with unit and a short text on what
  it means; nothing else about the twin is shown (D-034).
- **e. Personalization:** the Part 5 cold-start curve (Plotly, from stored numbers) with the
  honest paragraph beside it.
- **f. Input tiers:** a three-row table (full, no CGM with fingersticks, static only) with MAE at
  +30/+60/+120 min from Part 3 and the optimistic-fingerstick caveat.
- **g. Data-quality gate:** gate rule in plain words, the missing-data example, and the honest
  limits (detects gaps, not bias or drift; the 20 mg/dL-per-day drift coverage result).
- **h. Evidence:** six figures in a two-column grid with a plain-language caption each, then the
  limitations list including the failed dose-response test and the carb understatement.
- **i. Export:** a one-page PDF preview summary and "Export visit summary".

```
| Title (serif)                                              |
| One paragraph (max 72 characters wide)                     |
| [ figure or table, left ]      | [ caveat text, right ]    |
```


## 4. Principles

1. **Spend boldness in one place: the forecast chart.** It is the only large, framed, surfaced
   object in the product. It gets the most space, the only animation, the only pattern fills and
   the most careful typography. Everything else is plain text on paper.
2. **Keep everything else quiet.** No cards, no shadows, no gradients. Hairlines and whitespace
   separate content. One accent hue (damson) for interaction and the forecast.
3. **Say the decision in a sentence first, then show the evidence.** The serif decision sentence is
   always above the chart; the chart supports the sentence rather than replacing it.
4. **Honesty is part of the layout.** The what-if note, the data-quality limits and the disclaimer
   sit in fixed places, in body size, in ink colour. They are not hidden in tooltips or grey
   micro-text. A missing number is written "not available".
5. **Colour is never the only signal.** Every glucose state has a pattern and a label.
6. **Plain words, units on every number.** Sentence case, active voice, specific buttons ("Run
   what-if", "Export visit summary"). Errors say what went wrong and how to fix it. Empty states
   say what to do next.
7. **No decoration.** No emoji, no all-caps eyebrow labels, no middle-dot strings, no numbered
   markers, no arrows on buttons, no accent word in a headline.

## 5. Chart specification (the hero)

- Plot area about 16:7 at 1440 px and 16:9 at 820 px. x: last 6 hours of history and 2 hours of
  forecast. y: 40 to 300 mg/dL, fixed so a patient's trace does not rescale during replay.
- Zones as horizontal bands behind the data: low below 70 (hatch), target 70 to 180 (mint tint),
  high 180 to 250 (dots), very high above 250 (cross-hatch). Labels at the right edge.
- History: ink line, 2 px. "Now": a vertical rule labelled "Now" with the current value.
- Forecast: damson median line with markers at +30, +60 and +120 min; the 80% band drawn as a fan
  that widens toward +120. The band at each horizon is also printed as numbers in a text table
  under the chart, so hover is never the only way to read it.
- Meals: small triangles on the time axis, carbs in grams on hover and in the table.
- What-if: the baseline stays solid; the what-if is dashed damson with a lighter fan (ghost).
- Plotly modebar hidden. Loading state: the frame with the zone bands already drawn and the text
  "Loading forecast".


## 6. Review of this plan against "generic default"

I re-read the plan looking for anything I would produce for any dashboard. Item 2 is a change from a measured first choice; the others are defaults I started to reach for in my head and replaced before writing them into the plan.

1. **Cards in a grid.** The default overview is four metric cards (risk, trend, quality,
   confidence). Replaced by one serif sentence plus a
   confidence badge, and the profile as a plain two-column list. The only framed object left is
   the chart.
2. **Brand colour.** First choice was an indigo (`#4535A6`). It sat 20 CIELAB units from "low
   blue" under deuteranopia, and indigo is the default "tech" accent. Changed to damson
   `#43285E`: darker, at least 27.5 units from all four glucose colours, and reads as ink-plum.
3. **Typeface pairing.** One sans with bold headings is the default. Changed: a serif (Source
   Serif 4) only for titles and the decision sentence, so what the system says reads like a
   clinical note and controls read as controls.
4. **Status badges.** The default is red / amber / green pills. Changed: the confidence badge
   carries a word ("Low", "Moderate", "High") plus a bar count (1, 2, 3) on a plain ink outline.
   Glucose colours are not reused for confidence, so a confident forecast of a low is never
   confused with the "low" zone.
5. **Gauges.** Semicircle dials waste space and are hard to read to the percent. Changed to three
   horizontal bars with the number at the bar end, a 0 to 100 percent scale and the alert
   threshold marked as a tick.
6. **Evidence page.** A gallery of six identical image cards is the default. Changed to a
   two-column layout with each figure paired to a three-line caption, no frames, and the
   limitations as running text below.
7. **Motion.** The default is a fade-in on every block. Not used. One moment remains (the
   fan draw-in) plus the 200 ms ghost fade after a click.
8. **Navigation.** The default is an icon list. Changed to text only (no icon set to bundle),
   with the selected item shown by a bar and weight.

Open risks, flagged rather than hidden:
- Atkinson's digit width is not checked yet (no brotli module in the venv to read woff2). The
  fallback is in the Typefaces section; I will verify visually in Step 4.
- Streamlit's own widgets (select, slider) can be themed through `config.toml` and CSS but not
  rebuilt, so some widget details may stay close to default.
- Pattern fills use Plotly `fillpattern`. I will verify in a screenshot. If a pattern does not
  render, I will use a tint plus text labels and say so.
- The 250 mg/dL band is a display band only (see Colour).

## 7. Data the app will read (offline, precomputed by `make app-data`)

Nothing is computed from raw data at run time. `make app-data` writes a small `app/demo/` folder
(target under 25 MB; no raw or processed data committed):
- Real patients: out-of-fold forecasts, CV+ 80% bands, calibrated event probabilities, alert
  decisions and data-quality scores for chosen windows of a few CGMacros patients. Every real row
  comes from a model that never saw that patient. Shanghai patients have no meal or macro inputs,
  so they appear only on pages that do not need them.
- One SYNTHETIC patient from the Part 4 simulator, labelled "Synthetic" on every page.
- What-if results: precomputed per demo window (presets and walk on or off) from the Part 7
  engine, plus the stored Part 7 pass rates.
- SHAP values and template sentences for the same windows.
- Anything not computed shows "not available".

## 8. Questions for the owner at this checkpoint

1. Is a "very high" display band at 250 mg/dL acceptable (display only, no risk computed)?
2. Is "damson" plum plus Atkinson Hyperlegible and Source Serif 4 acceptable, or do you want a
   different direction before I build?
3. The what-if page can only offer pre-computed scenarios (presets and a few custom gram levels,
   walk on or off), because running the engine live needs the training data. Custom meals will be
   limited to a grid of carbs, protein, fat and fibre values. Is that acceptable?

