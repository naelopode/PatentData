import os
from pymongo import UpdateOne
from tqdm import tqdm
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
client = MongoClient("localhost", 29012)
db = client["test-database"]
collection_txt = db["collection-txt2"]
collection_json = db["collection-json"]

list_items = list(collection_json.find({'Country': 'US'}, {'_id': 1, 'Publication Number': 1, 'OCR_source':1}))

publication_dict = {str(doc['Publication Number']): doc['_id'] for doc in list_items}
publication_ocr_source = {str(doc['Publication Number']): doc['OCR_source'] if 'OCR_source' in doc else None for doc in list_items}
path = '/lhstdata1/patentdata/processed/us/text/6b-symspell-corrected'
unheard_of = []
updates = []
for filename in tqdm(os.listdir(path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as file:
            content = file.read()
            # Process the content as needed
            if str(filename[:-4]) not in publication_dict:
                unheard_of.append(filename[:-4]) #store txt file without a db match
                continue

            if ((str(filename[:-4]) in publication_dict) and ((publication_ocr_source[str(filename[:-4])] == 'google') or (publication_ocr_source[str(filename[:-4])] is None))):
                updates.append(UpdateOne({"_id": publication_dict[filename[:-4]]}, {"$set": {"OCR": content, "OCR_source": "tesseract06bsc"}})) #anotate which run in case of error
    if len(updates) > 500:
        collection_json.bulk_write(updates)
        updates = []

with open('/scratch/students/ndillenb/metadata/processing/unheard_of.pkl', 'wb') as f:
    pickle.dump(unheard_of, f)
