# Limitations, responsible AI and India's DPDP Act

## Limitations
1. **Small, non-Indian real cohorts.** CGMacros is 45 US adults; Shanghai T2DM is 100 Chinese adults.
   There is no Indian patient data in this project. Indian results are simulations.
2. **No prospective or clinical validation.** All results are retrospective.
3. **Transfer penalty.** A model trained on another population is not better than "repeat the last
   value" and needs local training or calibration (see README 9.4).
4. **Synthetic data trains nothing.** Adding simulated patients to training never helped, so the
   synthetic cohort is used only for stress tests and demonstration.
5. **What-if sizes are lower bounds.** A pre-registered dose-response test failed.
6. **Sensor faults.** Bias and drift are not detected by the data-quality score.
7. **Hypoglycaemia** evidence is limited.
8. **Federated learning is simulated** on open-cohort patients grouped into clinics. Formal
   differential privacy is not claimed because it is not usable with a handful of clinics.
9. **Sleep, HRV and step count** are not in the open real data; they are tested on a simulated cohort.

## How the project handles privacy (design, not certification)
- Only open, de-identified or synthetic data. No raw or processed data is committed.
- No phone numbers, emails or other contact details in the repository.
- Federated design: raw records stay at the clinic; only model updates are exchanged.

## Mapping to the Digital Personal Data Protection Act, 2023 (not legal advice)
GlucoTwin uses no real personal data of Indian patients. If it were deployed, the Data Fiduciary
(the hospital or clinic) would need the following. This table is a design checklist.

| DPDP principle | How the design supports it |
|---|---|
| Notice and consent for a stated purpose | The tool needs a clear purpose (glucose decision support); consent would be collected by the clinic, not by the model code. |
| Purpose limitation | Inputs are limited to glucose, meals, activity and a few EHR fields; nothing else is used. |
| Data minimisation | Federated design keeps raw records on site; the model uses a small set of fields. |
| Accuracy | Calibrated uncertainty, a data-quality page and visible warnings instead of silent output. |
| Storage limitation and erasure | The dashboard bundle is rebuilt from source; nothing is stored beyond a session. |
| Security safeguards | Not provided by this prototype; encryption, access control and audit logs would be needed. |
| Breach notification, data principal rights, grievance officer | Organisational duties of the hospital, outside the code. |
| Children's data | Out of scope; adults only. |

Health data in India is also affected by ABDM / NDHM health-data-management policy and the ICMR
ethical guidelines for AI in healthcare; both would need review before any clinical pilot.
