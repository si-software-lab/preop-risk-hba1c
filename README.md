#### *this project is under construction (12/7/2025):*
# Pre-Surgical Risk MGMT 
### Clinical Decision Support (CDS) Alerts 
#### Epic EHI pre-surgical lab value for HbA1c (Hemoglobin A1c)

---
- Inputs: a list of example EIDs

- Pulls: 10 core USCDI v1 data elements (de-identified placeholders)

- Source: Cosmos DB (FHIR-formatted source)

- Metric: % of patients with controlled A1C (an operational + payer quality target)

- Output: quick visualization clinicians recognize immediately

- Key Risk Value: In the Epic EHI Export schema the lab value for HbA1c (Hemoglobin A1c)

- NSQIP Risk Model for HbA1c serum lab values

- SMART-ON-FHIR to Epic Push (CommunicationRequest + Task)

#### App Capability Statement in a Table:

| Capability                   | Where              |
| ---------------------------- | ----------------------------- |
| Patient-level USCDI fields   | Cosmos JSON query             |
| Cohort selection by EID      | `ARRAY_CONTAINS(@eids…)`      |
| Clinically meaningful metric | Controlled diabetes           |
| Payer-relevant QI metric     | HEDIS-aligned glycemic control |
| Visualization                | Matplotlib bar chart          |



#
#
#

### About this Project
## LAYER 0: DATA & WORKFLOW GOVERNANCE

### Clinical AI Orchestration Model: SWIM (data architecture)

### SIIM Workflow Initiative in Medicine (SWIM)
- The Society for Imaging Informatics in Medicine (SIIM) created SWIM as a standards-focused initiative to improve interopereable clinical workflows for medical imaging, AI model integration, and edge-to-enterprise orchestration.
- It originally began as a blueprint tool showing how data moves through  

#### Why use SWIM to model clinical workflow orchestration?
- Inside regulated workflows, the SIIM Workflow Initiative in Medicine (SWIM) is the best practical model currently available, but i 



## LAYER 1: DATA-INFRASTRUCTURE LAYER


In computing, the "data-infrastructure" provides the storage, compute, pipelines, exchange method, and procedural logging.
s data stores, schemas, provenance lineage, and semantics that downstream analyses depend on:
- In this layer, we focus on "producers" that generate content to be consumed by components further downstream in the workflow.
- Analogy: Acquiring ingredients, and preparing/cooking the food for a fine-dining meal in the kitchen.




### Event Logging
### ATNA
### IHE-SOLE



#


## LAYER 2: DATA ARCHITECTURE


DATA-ANALYTIC LAYER

#
In computing, the "analytics service layer" (ASL) delivers computed insights and the end-user experience (UX):
- The ASL leverages applied data science techniques such as forecasting, predictive analyses, causal analysis icluding causal inference (CI), forecasting, and machine learning (ML). 
- This proccess derives insights and generates visualizations for decision makers in healthcare.  
- Analogy: Plating and serving the meal with final touches in the dining room.

 

### Risk Models
- What is a risk model? A risk model is an analytic artifact that uses features engineering within the data model. 
- For example, it relies on a statistical or ML construct that predicts the answer to a scientific question of interest.
- For this project, the risk model is the foundation of the Clinical Decision Support (CDS) layer nested within the ASL.
- The risk model itself is a consumer of the data architecture, not part of it. 

### Selected Risk Model Comparison Table
| Model                                           | Low-Risk Threshold | Medium-Risk Threshold | High-Risk Threshold | Usage                               |
| ----------------------------------------------- | -----------------: | --------------------: | ------------------: |-------------------------------------|
| **Simplified Diabetes Surgical Risk Model**     |             < 7.5% |              7.5–8.5% |              > 8.5% | NSQIP Risk                          |
| **Cardiometabolic Surgical Optimization Model** |               < 7% |                  7–8% |                > 8% | Bariatric / high-risk cardiovascular |









