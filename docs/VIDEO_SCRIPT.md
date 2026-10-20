# GlucoTwin — demo video script

**Target length: 21 to 22 minutes. The organisers require a MINIMUM of 20 minutes.**
Twenty is a floor, not a target, so this script is written to land comfortably above it. Do not
rush. If you finish under twenty minutes, go back and re-record — an under-length video risks the
submission regardless of how good the work is.

**How to use this.** `[SCREEN]` says what should be visible. `[SAY]` is narration — read it, don't
improvise. The section timings are computed from the word counts at **132 words per minute**, which
is an unhurried presenting pace. Pause for a beat at each section break; the timings allow for it.

**All figures in this script are final.** They are read off the served tuned forecaster (D-072a)
and match the README and the slides. Nothing here is pending.

**Format.** The organisers allow the video to be recorded as a narrated PowerPoint presentation or
uploaded as an unlisted YouTube video. This script assumes you narrate over the deck and cut to the
live console and the Streamlit app for the two demo sections.

**Before recording**
- Close every other window. Hide bookmarks. Set display scaling so text is readable at 1080p.
- Have open and pre-loaded, in this tab order: (1) the slide deck, (2) the web console,
  (3) the Streamlit app (`make app`), (4) the GitHub repository.
- Pick **one** patient and use that same patient throughout. Choose one whose forecast shows a
  clear rise, so the what-if lab has something to change.
- Record at 1080p. Audio matters more than video: use a headset mic, not the laptop mic.
- Do a 30-second test recording first and listen back before committing to the full take.

---

## 0:00 – 1:44 · Opening

`[SCREEN]` Slide 1 — title.

`[SAY]`
> Type 2 diabetes is managed in appointments weeks or months apart. In that appointment a
> clinician sees an HbA1c, a few lab values, and whatever the patient remembers. Meanwhile the
> patient has generated three thousand glucose readings, a continuous record of their heart rate
> and movement, and every meal they ate — and nobody has time to read any of it.
>
> GlucoTwin is a patient-specific digital twin for type 2 diabetes. It fuses the static health
> record with the dynamic streams — continuous glucose, heart rate, activity, sleep and meals — and
> produces three things a clinician can act on. Where glucose is heading in the next two hours,
> with an honest statement of uncertainty. The calibrated probability of going above 180 or below
> 70. And the answer to "what if" — what happens if this patient eats less rice, or walks for half
> an hour after lunch.
>
> I'll show you the system working, then the evidence behind it, and then I'm going to be straight
> with you about what it cannot do. I would rather you heard the limitations from me than found
> them yourself.
>
> One thing to say at the outset about scope. The brief asks for one chronic condition, and this is
> type 2 diabetes only. Every design decision in the project was made for that population, and
> nothing here is a general-purpose health model wearing a diabetes label.

---

## 1:44 – 3:30 · What the brief asked for

`[SCREEN]` Slides 2 and 3.

`[SAY]`
> The brief asked for four things. One condition — type 2 diabetes only, and every design decision
> is made for that population. A static health record fused with dynamic wearable data — I'll show
> where each stream comes from and how much each contributes, including the ones that contribute
> very little, because that is also a result. A working model that predicts an adverse event — the
> system forecasts at thirty, sixty and a hundred and twenty minutes and converts that into a
> calibrated probability. Calibrated is the important word: when it says forty per cent, that
> happens about forty per cent of the time. And a clinician-facing dashboard — there are two, and
> you'll see both.
>
> Everything runs on open or clearly-labelled synthetic data. No real patient record is in this
> repository, and none ever was.
>
> The brief also names the two streams it wants fused, so let me be explicit. The static stream is a
> simulated electronic health record: demographics, past diagnoses, lab results, medications and
> genetic markers, in a schema compatible with Synthea. The dynamic stream is time series from
> consumer wearables and sensors: heart rate variability, continuous glucose, sleep stages and step
> count. All four of the dynamic channels the brief lists are present. And the worked example the
> brief gives — predicting a glucose spike two hours in advance — is exactly the +120 minute
> forecast and the adverse-event probability you are about to see.

---

## 3:30 – 5:10 · The data

`[SCREEN]` Slide 4.

`[SAY]`
> Three sources, and I'll be precise about which is real and which is simulated.
>
> CGMacros, from PhysioNet — forty-five adults with real continuous glucose, real Fitbit heart
> rate and activity, meal macronutrients and labs. The main development cohort.
>
> The Shanghai type 2 dataset, from figshare — a hundred patients with glucose, meals and labs. A
> genuinely separate population: different country, different diet, different clinical practice.
> Every headline number in this video is reported in both.
>
> And a synthetic cohort from a simulator in this repository: glucose, heart rate, steps, sleep
> stages and heart rate variability, with Indian meals whose carbohydrate comes from the Indian
> Food Composition Tables of 2017, and insulin sensitivity lowered using published clamp data.
>
> Why generate data at all? Two reasons. The brief explicitly permits synthetic data and names
> tools like Synthea for the record side, so it is within the sandbox rules. And more importantly:
> no open dataset anywhere pairs real continuous glucose with real sleep staging and heart rate
> variability in type 2 patients. That dataset does not exist. I looked hard. What I did about it
> turned into one of the more interesting results in the project, and I will come back to it.
>
> Everything simulated is labelled SIMULATED everywhere it appears — code, dashboards, README and
> figures. You will never have to guess which is which.

---

## 5:10 – 6:46 · Architecture

`[SCREEN]` Slide 5, then `docs/architecture.pdf` full screen.

`[SAY]`
> This is the whole system on one page.
>
> At the top, the four input streams. Below that, harmonisation — every cohort on one five-minute
> timeline with a shared schema. Two rules matter more than they look. Every feature is causal:
> none may use information from after the moment of prediction, enforced in code and covered by
> tests. And the splits are patient-grouped — training, validation, calibration and holdout
> patients never overlap. Those two are the difference between a real result and a leak.
>
> Then the model layer. The served forecaster is a tuned gradient-boosted model predicting the
> change from the current value, trained on absolute error — the metric it is scored on. Alongside
> it sits a mechanistic twin: a glucose–insulin differential equation system fitted per patient,
> integrated with Runge–Kutta, with an extended Kalman filter tracking the state. That is what
> makes this a twin rather than just a forecaster — a physiological state you can intervene on.
>
> Below that, calibration — the layer I'd point to if you asked what separates this from a weekend
> project. A raw model output is a number with no stated reliability; this turns it into an eighty
> per cent interval and a calibrated event probability. Then the clinical outputs, and the two
> interfaces. Let's look at them.

---

## 6:46 – 10:29 · Live demo — the clinician console

`[SCREEN]` The live web console, patient list.

`[SAY]`
> This is the console, live in a browser — no install, no sign-in. It opens on the patient list,
> because that is how a clinic day actually starts: not with one patient, but with the question of
> which one needs attention first. Rows are ordered by the calibrated chance of a dangerous high.

`[SCREEN]` Open your chosen patient — the "Now and next" tab.

`[SAY]`
> Here is one patient. Recent history on the left, the forecast at thirty, sixty and a hundred and
> twenty minutes with the eighty per cent band around it. The band widens with the horizon, which
> is what honesty looks like — the model is less sure about two hours from now, and says so.
>
> Underneath, one plain sentence, and the calibrated probability of crossing the thresholds. Note
> what it does not do: it does not tell the clinician to do anything, and it does not suggest a
> dose. The clinician remains the decision maker, and that is written into the project rules.

`[SCREEN]` The What-if lab. Replace a logged meal with an Indian preset, then toggle the walk.

`[SAY]`
> This is the what-if lab, and it is what makes the twin useful rather than just informative.
> Historical data can never answer this, because the alternative never happened — only a model you
> can intervene on can.
>
> Here is a meal this patient actually ate. I replace it with one of sixteen Indian presets, so the
> question is asked in plates and katoris rather than grams, and add a thirty-minute walk. The
> forecast updates both times.
>
> One thing is guaranteed rather than hoped for: the served model carries a monotonicity
> constraint, so less carbohydrate can never predict a higher peak. That check passes a hundred per
> cent of the time; the unconstrained version passes twenty-seven.
>
> And look at what is printed directly beneath the result — the limits of the answer, next to the
> answer. It cannot know insulin doses, illness or stress. On eighty-seven matched real breakfasts
> the models predict about nineteen of the thirty-five milligrams per decilitre observed, so every
> magnitude is shown as a lower bound. That is on screen beside the number, every time.

`[SCREEN]` Glucose report, health record, sleep and wearables, visit summary, access.

`[SAY]`
> The glucose report follows the 2019 international consensus — the ambulatory glucose profile,
> time in range against the standard thresholds, and the glucose management indicator. This is
> deliberately conventional. A clinician should not have to learn a new vocabulary to read it.
>
> The health record tab is the static stream in full: demographics, diagnoses, medications, kidney
> function and genetic markers — the simulated EHR the brief asks for. And here is sleep, heart
> rate variability and step count, presented as clinical context rather than as forecast inputs.
> That is a deliberate choice and it was driven by a measurement, not a preference. I will show you
> the measurement in the results.
>
> The visit summary fits the two minutes a clinician actually has. Access is role-gated and every
> access is logged — a demonstration of the governance layer, not a production authentication
> system. And a second view, Evidence, carries every cohort-level result with nine figures, so a
> reviewer can check a claim without leaving the page.

---

## 10:29 – 11:10 · Live demo — the Streamlit application

`[SCREEN]` Streamlit app. Move through pages as named.

`[SAY]`
> The console is the front door. The full application carries the pages a developer or reviewer
> needs, and reads the same bundle, so the two can never disagree on a number.
>
> Twin state shows the fitted physiological state of the mechanistic model. Personalisation shows
> how the model improves as it sees more of this patient. Input tiers answers "what if this clinic
> doesn't have all the data". Data quality flags gaps in the streams. Evidence collects every
> result in one place, and a PDF export puts something in the patient's file.

---

## 11:10 – 16:36 · Results

`[SCREEN]` Slides 8 through 16, in order.

`[SAY]`
> Every number has a confidence interval bootstrapped over patients, not over time windows — two
> readings from one person are not two independent facts.
>
> Accuracy. Ten point nine milligrams per decilitre at thirty minutes, fifteen point seven at
> sixty, twenty at two hours. Nineteen per cent better than repeating the last reading, which is
> the clinical default, and it replicates at fifteen point five in the second cohort. On the Clarke
> Error Grid, ninety-nine point six per cent of forecasts land in zones A or B.
>
> Error and interval coverage are reported per subgroup — by HbA1c band, age and sex — as well as
> pooled, so a model that works on average but fails for one group cannot hide inside a single
> number. Robustness is tested rather than assumed: sensor faults are injected into the test data
> and every causal feature recomputed, and a data-quality score travels with each forecast.
>
> Which inputs matter? I removed each group of features and retrained. Meals are worth nine tenths
> of a milligram per decilitre across the day, and two point zero in the run-up to a high, where
> the clinician is actually deciding. Wearables are worth nought point one two. Four of seven
> feature groups are worth nothing distinguishable from zero. I report that because a result that
> contradicts the expected story is worth more than one that confirms it.
>
> Which model? Four families built and scored: tuned gradient boosting, a mechanistic twin, a GRU
> sequence model, and a physics-constrained hybrid. The sequence model came out indistinguishable
> from boosting; the physics hybrid came out three and a half milligrams per decilitre worse. So
> the simplest one is served, and the twin is kept as an independent engine cross-checking every
> what-if. Thirty-seven configurations in all, each in the repository beside the number that ruled
> it out.
>
> Uncertainty. An eighty per cent band is only honest if it contains the truth eighty per cent of
> the time, so I built four conformal methods and judged them on that alone. Three fail somewhere:
> two drop to seventy-eight and seventy-six per cent in the second cohort, and the adaptive method
> hits its target by returning an infinite interval up to forty-four per cent of the time —
> technically perfect, useless at the bedside. The one I serve holds everywhere, at both the eighty
> and ninety-five per cent levels.
>
> Adverse events. Two hours ahead — the competition's own case — the area under the ROC curve for
> crossing above 180 is zero point eight two on Shanghai, zero point seven seven on CGMacros, and
> probabilities are calibrated to within two and a half points. At the alert load a clinic carries
> today, the model catches eighty-three per cent of episodes against fifty-nine for the alert they
> run now.
>
> Personalisation — the part that makes this a twin. Seven days of one patient's own data takes
> the error from sixteen point one to fourteen point nine, still improving at the end of the week,
> and a second method pays from the first reading. Training on that patient alone reaches eighteen
> point three — worse than not personalising at all.
>
> Deployment. Moving a model to the other country costs three point four milligrams per decilitre,
> and three days of local data closes much of that — the recipe the console uses. Federated
> training, where records never leave the building, costs nine hundredths of a milligram against
> pooling everything centrally.
>
> And sleep — the result I would most like you to remember. No real data can answer this, so I
> simulated nine thousand patient-days, with sleep affecting insulin sensitivity at the strength
> the literature reports and then at five times that. Adding sleep and heart rate variability
> changes the forecast by at most nine hundredths of a milligram per decilitre. The effect is
> unmistakably in the data — inflating it moves mean glucose and time above 180 — the model just
> does not need the channel to see it, because glucose is downstream of sleep. By the time a
> reading exists, last night is already inside it.
>
> Turn that around and it is a statement about the field: a study designed to find this effect has
> to be larger than nine thousand patient-days — larger than any open CGM cohort in existence. That
> is a sample-size number anyone attempting this can use, and I have not found it published.

---

## 16:36 – 18:34 · How the work was done

`[SCREEN]` Slide 17, then `docs/DECISIONS.md`. Scroll to show the volume of entries.

`[SAY]`
> A minute on method, because I think it is the real contribution.
>
> A model can always be made to look good by picking the comparison once you have seen the answer.
> So every such choice was recorded first, in a plain-text decision log: seventy-four numbered
> entries, seventy-two of them written before the work they govern, each saying what would be done,
> why, and what would count as it having worked.
>
> Here is one. It fixed the adoption rule and a seventy-five per cent coverage floor before the
> tuned model was ever scored, so shipping it could not be decided by looking at the result.
> Coverage came back at eighty point three or better everywhere. And here is one where a check fell
> outside its pre-registered range — recorded anyway, with the departure written up.
>
> Every result file carries the commit, the data hash and the seed that produced it — thirty-three
> out of thirty-three. Two hundred and twelve unit tests run from one command. Every figure in the
> deck is regenerated from those files rather than redrawn, so the slides and the repository cannot
> drift apart.
>
> Everything regenerates from source with a single command per stage. Continuous integration runs
> linting, formatting and the full test suite on every push, and it is green. There is a Docker
> image, a datasheet describing the data, and a model card describing the model and its intended
> use. If you want to check any number in this video, the repository tells you which file it came
> from and which decision governed the run that produced it.

---

## 18:34 – 20:47 · Limits, and what comes next

`[SCREEN]` Slide 19.

`[SAY]`
> The limits, stated by me rather than discovered by you, each paired with the work that closes
> it.
>
> A hundred and forty-five real patients across two cohorts. That is small, and none are in India;
> there is no open Indian CGM dataset and I looked hard. What closes it is a fifty to a hundred
> patient study in an Indian clinic, and the three-day calibration recipe is already measured.
>
> The Indian, sleep and federated results are simulations — they test design, not clinical
> performance. Running the federated protocol across two real sites closes that; the code is
> written and the privacy accounting is in place.
>
> The what-if magnitudes are lower bounds, and the interface says so every time. The quality score
> does not catch sensor bias — a drift detector closes that. Interval coverage is not uniform
> across subgroups — group-conditional conformal prediction closes that, and it is code, not data.
> And the hypoglycaemia evidence is thin: eight CGMacros patients ever went below seventy.
>
> And the one that matters most: this has not been prospectively validated. It is decision support,
> not a medical device, and nothing here substitutes for a planned study and a regulatory path.
>
> What would make this stronger, in order. Access to AI-READI, which as far as I can find is the
> only dataset pairing real continuous glucose with real sleep staging and heart rate variability
> in type 2 patients — that request is in. A prospective pilot in a clinic. And a federated
> deployment across real sites, which the simulation says should work and only a deployment can
> prove.
>
> What I am looking for is one Indian diabetes clinic willing to run CGM and a meal log for a
> month, and a second site to federate with. That is the whole ask.

---

## 20:47 – 21:49 · Close

`[SCREEN]` Slide 20.

`[SAY]`
> Six findings, each measured on held-out patients in two independent cohorts, each carrying a
> confidence interval, each reproducible from the repository.
>
> A forecast that holds in two countries. Uncertainty that keeps its promise at two confidence
> levels. Calibrated risk instead of alarms, catching eighty-three per cent of episodes at the
> alert load a clinic already carries. A twin that measurably adapts to one person over seven days.
> A counterfactual engine with a guarantee that is enforced rather than hoped for. And a route to
> deployment in India where the entire cost of never moving a patient record is nine hundredths of
> a milligram per decilitre.
>
> The console is live right now. The code, the documentation, every decision and every number is in
> the repository — including the results that did not go my way. Thank you for watching.

---

## After recording

- Watch it back once, start to finish, at normal speed.
- Confirm total length is **over 20 minutes** — that is a hard requirement, not a target. If it
  lands under, re-record rather than submitting short.
- Upload as unlisted on YouTube, then paste the link into README section 11, replacing
  `[VIDEO LINK]`.
- Check the link works in a private browser window before submitting.
