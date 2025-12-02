import os
import json
import requests
import pandas as pd
from azure.cosmos import CosmosClient
import matplotlib.pyplot as plt

#  cosmos cnx
COSMOS_URL = os.getenv(COSMOS_URL)  # https://your-account.documents.azure.com:443/
COSMOS_KEY = os.getenv(COSMOS_KEY)  # YOUR_COSMOS_KEY
DATABASE_ID = os.getenv(DATABASE_ID)  # PatientDB
CONTAINER_ID = os.getenv(CONTAINER_ID)  # USCDI
DB = os.getenv(EHI_DB)  # EHI_DB

client = CosmosClient(COSMOS_URL, COSMOS_KEY)
db = client.get_database_client(DB)

# containers in cosmos aligned to uscdi / epic data sources
c_pat = db.get_container_client("PATIENT")
c_labs = db.get_container_client("NSQIP_PREOP_LABS")
c_vitals = db.get_container_client("PREOP_VITALS")

#  pre-op cohort query (day-of-surgery) 
sql_cohort = """
    select p.patient_id,
           p.encounter.id AS encounter_id,
           p.birthdate,
           p.gender,
           p.surgical_risk
    from EDW.EPIC.PATIENT as p
    where p.encounter.preop_date = GetCurrentDate()
"""
pat = list(c_pat.query_items(query=sql_cohort, enable_cross_partition_query=True))
df_pat = pd.DataFrame(pat)

#  most recent pre-op a1c 
sql_labs = """
    select l.patient_id,
           l.labs.a1c.value AS a1c,
           l.labs.a1c.effectiveDateTime AS lab_time
    from l
    where l.labs.a1c IS NOT NULL
"""
labs = list(c_labs.query_items(query=sql_labs, enable_cross_partition_query=True))
df_labs = pd.DataFrame(labs)

#  day-of-surgery vitals 
sql_vitals = """
    select v.patient_id,
           v.vitals.bp.systolic AS sbp,
           v.vitals.bp.diastolic AS dbp,
           v.vitals.bmi AS bmi,
           v.time AS vitals_time
    from v
    where v.time = GetCurrentDate()
"""
vitals = list(c_vitals.query_items(query=sql_vitals, enable_cross_partition_query=True))
df_vitals = pd.DataFrame(vitals)

#  merge all data 
df = df_pat.merge(df_labs, on="patient_id", how="left") \
           .merge(df_vitals, on="patient_id", how="left")

#  NSQIP surgical diabetes risk categories 
def classify_risk(a1c):
    if a1c is None:
        return "Unknown"
    if a1c < 7.5: return "Low"
    if 7.5 <= a1c <= 8.5: return "Medium"
    return "High"

df["risk_category"] = df["a1c"].apply(classify_risk)

#  hybrid workflow action 
def action_for_risk(risk):
    if risk == "High":
        return "Block"    # HITL stop
    if risk == "Medium":
        return "Warn"     # Banner only
    return "None"

df["workflow_action"] = df["risk_category"].apply(action_for_risk)


# return-to-epic fhir alerting
FHIR_BASE = "https://openepic.example.org/fhir"
TOKEN = os.getenv(OAUTH_BEARER)  # YOUR_OAUTH_BEARER

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/fhir+json"
}

def send_alert(row):
    if row["workflow_action"] == "None":
        return

    patient_ref = f"Patient/{row['patient_id']}"
    encounter_ref = f"Encounter/{row['encounter_id']}"

    #  always send a communication request 
    comm = {
        "resourceType": "CommunicationRequest",
        "status": "active",
        "subject": {"reference": patient_ref},
        "encounter": {"reference": encounter_ref},
        "payload": [{
            "contentString": f"Pre-op A1C {row['a1c']}% ({row['risk_category']})."
        }],
        "reasonCode": [{"text": "Glycemic Surgical Risk"}]
    }
    requests.post(f"{FHIR_BASE}/CommunicationRequest",
                  headers=headers,
                  data=json.dumps(comm))

    #  additional task for block events 
    if row["workflow_action"] == "Block":
        task = {
            "resourceType": "Task",
            "status": "requested",
            "intent": "order",
            "description": "High surgical diabetes risk — HITL sign-off required.",
            "for": {"reference": patient_ref},
            "encounter": {"reference": encounter_ref},
            "priority": "stat"
        }
        requests.post(f"{FHIR_BASE}/Task",
                      headers=headers,
                      data=json.dumps(task))

df.apply(send_alert, axis=1)


# visualization for clinical huddle in the OR
risk_counts = df["risk_category"].value_counts()

plt.figure(figsize=(6, 4))
risk_counts.plot(kind="bar")
plt.title("Preoperative A1C Risk Category — NSQIP Surgical Diabetes Thresholds")
plt.ylabel("Patient Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(df[["patient_id", "encounter_id", "a1c", "risk_category", "workflow_action"]])
