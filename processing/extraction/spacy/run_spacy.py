import os
from pymongo import UpdateOne
from tqdm import tqdm
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
import spacy
import json
import random

client = MongoClient("localhost", 29012)
db = client["test-database"]
collection_txt = db["collection-txt2"]
collection_json = db["collection-json"]
COUNTRY = "GB" #Select the country you want to process
nlp = spacy.load("./spacy_files/output_gb/model-best") #Load the trained spaCy model
for collection in [collection_txt, collection_json]:
    print(f"Processing collection {collection.name}")
    all_items = list(collection.find({'Country': COUNTRY, "OCR": {"$exists": True}}, {'_id':1, 'OCR':1}))
    count_skipped = 0
    updates = []
    for item in tqdm(all_items):
        text = item['OCR']
        if text is None or text == '':
            count_skipped += 1
            continue
        names = []
        doc = nlp(text)
        predicted = {
            "Title":None,
            "Application_Date":None,
            "Publication_Date":None,
            "Applicants":[],
        }
        for ent in doc.ents:
            if ent.label_ == "TITLE":
                predicted["Title"] = ent.text
            elif ent.label_ == "APPLICATION_DATE":
                predicted["Application_Date"] = ent.text
            elif ent.label_ == "PUBLICATION_DATE":
                predicted["Publication_Date"] = ent.text
            elif ent.label_ == "APPLICANT":
                    if not any(fuzz.token_sort_ratio(ent.text.lower(), name.lower()) >= 70 for name in predicted["Applicants"]):
                        predicted["Applicants"].append(ent.text)
        updates.append(UpdateOne({'_id': item['_id']}, {
            "$set": {
                "spacy": predicted
            }
        }))
        if len(updates) >= 4000:
            collection.bulk_write(updates)
            updates = []
    if updates:
        collection.bulk_write(updates)
        updates = []
    print(f"Skipped {count_skipped} items")