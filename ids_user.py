import csv

from bson.int64 import Int64
import pandas as pd
from pymongo import MongoClient

mongo_client = MongoClient("mongodb://127.0.0.1:27020/",
                           username="twitter",
                           password="twitter",
                           authSource="twitter",
                           authMechanism="SCRAM-SHA-1")

db = mongo_client["twitter"]
c = db["QCPS_2"]

df = pd.read_csv("/ipazianas/pasquini/output_graph_analysis/f6e06c4ca1a011efa50708f1eaf4fe18/user_id", sep=",", header=None)
user_ids = df.iloc[:, 0].tolist()
user_ids_int64 = [Int64(x) for x in user_ids]

seen_ids = set()
csv_file = "/ipazianas/pasquini/output_graph_analysis/id_user_name.csv"

with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["user_id", "screen_name"])
    writer.writeheader()

    for user_id in user_ids:
        doc = c.find_one(
            {"user.id": user_id},
            {"user.id": 1, "user.screen_name": 1}
        )
        if doc:
            uid = doc["user"]["id"]
            if uid not in seen_ids:
                seen_ids.add(uid)
                writer.writerow({
                    "user_id": uid,
                    "screen_name": doc["user"]["screen_name"]
                })
