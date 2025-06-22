from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import bisect
import dateparser
from classification_predictor import Predictor_Class
"""
This script aggregates data from two MongoDB collections (one for text and one for JSON) into a new collection.
It processes each document to extract and format key information, including publication and application dates,
key people, and classifications. It also predicts missing publication dates for German patents using a custom predictor
and fills in missing classifications for UK patents using a subclass predictor. This predictor uses a pre-trained model and a mlb.plk
Please first run the multiclass_pytorch.py script to generate the mlb.pkl file and the pre-trained model.
You can run with PREDICTOR = False to skip the prediction step and just aggregate the data, to train the predictor.
"""
PREDITOR = True
class Predictor:
    def __init__(self, collection_txt, collection_json, query):
        self.collection_txt = collection_txt
        self.collection_json = collection_json
        self.x = []
        self.y = []
        self.size_range = 5
        self.load_data(query)

    def load_data(self, query):
        x = []
        y = []
        for item in self.collection_txt.find({"Country":"DE"}, {'_id':1, 'ID':1, query:1}):
            if query in item and item[query] is not None:
                x.append(int(item[query][:4]))
                y.append(int(item['ID'][2:-1]))
        for item in self.collection_json.find({"Country":"DE"}, {'_id':1, 'ID':1, query:1}):
            if query in item and item[query] is not None:
                x.append(int(item[query][:4]))
                y.append(int(item['ID'][2:-1]))
        x_sorted_y, y_sorted = zip(*sorted(zip(x, y), key=lambda pair: pair[1]))
        self.x = list(x_sorted_y)
        self.y = list(y_sorted)

    def predict(self, id_value):
        idx = bisect.bisect_left(self.y, id_value) #locate where it would fit in the sorted list
        if idx-self.size_range > 0 and idx+self.size_range < len(self.x) and all(list(self.x[idx-i] == self.x[idx] for i in range(-self.size_range,self.size_range))): #test the values in both direction for 5 matches
            return self.x[idx] if idx < len(self.x) else None
        else:
            return None
        
    def predict_sample(self, id_value): #This is for validation purpose, we delete the value from the list before prediction
        true_idx = bisect.bisect_left(self.y, id_value)
        x_temp = self.x[true_idx-100:true_idx] + self.x[true_idx+1:true_idx+100]
        y_temp = self.y[true_idx-100:true_idx] + self.y[true_idx+1:true_idx+100]
        idx = bisect.bisect_left(y_temp, id_value)
        return x_temp[idx] if idx < len(x_temp) else None
    
def parse_date_with_language(date_str, language):
    supported_languages = ['fr', 'de', 'en']
    if date_str is None:
        return None

    if language not in supported_languages:
        raise ValueError(f"Unsupported language: '{language}'. Use one of {supported_languages}.")

    try:
        dt = dateparser.parse(date_str, languages=[language])
        return dt
    except Exception as e:
        print(f"Error parsing date: {e}")
        return None

def process_doc(doc, dict_pn2name, dict_pn2date, germandate_predictor, subclass_preditor):
    new_data = {
        'ID': doc['ID'],
        'Title': doc['Title'] if 'Title' in doc else None,
        'Country': doc['Country'],
        'Classification': doc['Classification'] if 'Classification' in doc else None,
        'key_people': [],
        'Publication Date': None,
        'Application Date':None,
    }
    if 'OCR' in doc:
        new_data['OCR'] = doc['OCR']
    key_people = []
    if 'Inventor' in doc and doc['Inventor'] is not None:
        for person in doc['Inventor']:
            key_people.append(person)
    if 'Applicant' in doc and doc['Applicant'] is not None:
        for person in doc['Applicant']:
            key_people.append(person)
    if len(key_people)==0 and 'spacy' in doc:
        for person in doc['spacy']['Applicants']:
            key_people.append(person)
    if len(key_people)==0 and doc['ID'][0:2]+'-'+doc['ID'][2:-1]+'-'+doc['ID'][-1:] in dict_pn2name.keys():
        key_people.append(dict_pn2name[doc['ID'][0:2]+'-'+doc['ID'][2:-1]+'-'+doc['ID'][-1:]])
    new_data['key_people'] = list(set(key_people))  # Remove duplicates

    #Publication Date
    if 'C_Publication Date' in doc and doc['C_Publication Date'] is not None and doc['C_Publication Date'] != '':
        new_data['Publication Date'] = doc['C_Publication Date']
    elif doc['ID'][0:2]+'-'+doc['ID'][2:-1]+'-'+doc['ID'][-1:] in dict_pn2date.keys():
        publication_date = dict_pn2date[doc['ID'][0:2]+'-'+doc['ID'][2:-1]+'-'+doc['ID'][-1:]]
        if isinstance(publication_date, float):
            publication_date = str(int(publication_date))
        new_data['Publication Date'] = f"{publication_date[:4]}-{publication_date[4:6]}-{publication_date[6:]}"
    if doc['Country'] == 'DE' and (doc['ID'][0:2]+'-'+doc['ID'][2:-1]+'-'+doc['ID'][-1:] not in dict_pn2date.keys() or new_data['Publication Date'] is None or new_data['Publication Date']==''):
        publication_year = germandate_predictor.predict(int(doc['ID'][2:-1]))
        if publication_year:
            new_data['Publication Date'] = f"{publication_year}-01-01"
        else:
            new_data['Publication Date'] = None

    #Application Date
    if 'C_Application Date' in doc and doc['C_Application Date'] is not None and doc['C_Application Date'] != '':
        new_data['Application Date'] = doc['C_Application Date']
    elif doc['ID'][0:2]+'-'+doc['ID'][2:-1]+'-'+doc['ID'][-1:] in dict_pn2date.keys():
        publication_date = dict_pn2date[doc['ID'][0:2]+'-'+doc['ID'][2:-1]+'-'+doc['ID'][-1:]]
        if isinstance(publication_date, float):
            publication_date = str(int(publication_date))
        new_data['Application Date'] = f"{publication_date[:4]}-{publication_date[4:6]}-{publication_date[6:]}"
    #Title
    if 'Title' not in doc or doc['Title'] is None or doc['Title'] == '':
        if 'spacy' in doc and 'title' in doc['spacy'] and doc['spacy']['title'] is not None and doc['spacy']['title'] != '':
            new_data['Title'] = doc['spacy']['title']
    if subclass_preditor is not None:
        if doc['Country'] == 'GB' and 'Title' in new_data and new_data['Title']!='' and new_data['Title'] is not None and ('Classification' not in doc or doc['Classification'] is None or len(doc['Classification'])==0 or doc['Classification'] == '' or not any(doc['Classification'])) :
            predicted_class,_ = subclass_preditor.predict(new_data['Title'])
            dict_classes = {}
            for i, class_name in enumerate(predicted_class):
                dict_classes[f"EP-CPCI-{i+1}"]={'section': class_name[0], 'class': class_name[1:3], 'subclass': class_name[-1], 'generating-office':'Predictor-ndillenb'}
            new_data['Classification'] = [dict_classes]
    # Return data
    return new_data

def __main__():
    # Init everything
    client = MongoClient("localhost", 29012)
    db = client["test-database"]
    collection_txt = db["collection-txt2"]
    collection_json = db["collection-json"]
    subclass_predictor = Predictor_Class() if PREDITOR else None


    all_pc_name = list(db['PatentCity'].find({},{'publication_number':1, 'name_text':1, 'publication_date':1}))
    dict_pn2name = {item['publication_number']: item['name_text'] for item in all_pc_name}
    dict_pn2date = {item['publication_number']: item['publication_date'] for item in all_pc_name}

    collection_agg = db["aggregated-data4"]
    insert_data = []

    predict_germandate = Predictor(collection_txt, collection_json, 'C_Publication Date')
    
    for cursor, size in [(collection_json.find({}),collection_json.count_documents({})), (collection_txt.find({}),collection_txt.count_documents({}))]:
        print("Processing dataset")
        with tqdm(total=size) as pbar:
            for doc in cursor:
                new_data = process_doc(doc, dict_pn2name, dict_pn2date, predict_germandate, subclass_predictor)
                insert_data.append(new_data)
                if len(insert_data) >= 1000:
                    collection_agg.insert_many(insert_data)
                    insert_data = []
                pbar.update(1)
            if len(insert_data) > 0:
                collection_agg.insert_many(insert_data)
                insert_data = []

__main__()