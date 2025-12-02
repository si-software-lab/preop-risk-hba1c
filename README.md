# preop-risk-hba1c
### Clinical Decision Support Alert: Epic EHI PREOP lab value for HbA1c (Hemoglobin A1c)

---
**this project is under construction:**
- Inputs: a list of example EIDs

- Pulls: 10 core USCDI v1 data elements (de-identified placeholders)

- Source: Cosmos DB (FHIR documents)

- Metric: % of patients with controlled A1C (an operational + payer quality target)

- Output: quick visualization clinicians recognize immediately

- Key Risk Value: In the Epic EHI Export schema the lab value for HbA1c (Hemoglobin A1c)

- NSQIP Risk Model for 

- SMART-ON-FHIR to Epic Push (CommunicationRequest + Task)

#

### Current Project Trajectory
| Capability                   | Where it’s shown               |
| ---------------------------- | ------------------------------ |
| Patient-level USCDI fields   | Cosmos JSON query              |
| Cohort selection by EID      | `ARRAY_CONTAINS(@eids…)`       |
| Clinically meaningful metric | Controlled diabetes            |
| Payer-relevant QI metric     | HEDIS-aligned glycemic control |
| Visualization                | Matplotlib bar chart           |

#


### Risk Models
| Model                                           | Low-Risk Threshold | Medium-Risk Threshold | High-Risk Threshold | Usage                               |
| ----------------------------------------------- | -----------------: | --------------------: | ------------------: |-------------------------------------|
| **Simplified Diabetes Surgical Risk Model**     |             < 7.5% |              7.5–8.5% |              > 8.5% | NSQIP Risk                          |
| **Cardiometabolic Surgical Optimization Model** |               < 7% |                  7–8% |                > 8% | Bariatric / high-risk cardiovascular |
