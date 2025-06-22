from pymongo import MongoClient
import os
import json
from tqdm import tqdm
import datetime
import pandas as pd
import re
from collections import defaultdict
from datetime import datetime


client = MongoClient("localhost", 29012)
db = client["test-database"]
collection = db["collection-txt2"]

def parse_patent_string(data: str):
    result = defaultdict(list)
    current_section = None
    pattern = re.compile(r'(\w+)=["\'](.*?)["\']')

    #detect date fields (YYYY-MM-DD)
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')

    # Tokenize 
    tokens = re.split(r'\s{2,}', data)  # Split by two or more spaces

    for token in tokens:
        # Detect section name and preserve it
        if ":" in token:
            parts = token.split(":", 1)
            current_section = parts[0].strip()
            token = parts[1].strip()

        # Extract key-value pairs
        matches = pattern.findall(token)
        entry = {}
        
        for key, value in matches:
            # Convert dates to YYYY-MM-DD format
            if date_pattern.match(value):  
                try:
                    value = datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
                except ValueError:
                    pass  # In case of an unexpected format, keep the original

            entry[key] = value

        if entry:
            if current_section:
                result[current_section].append(entry)
            else:
                result.update(entry)  # Direct assignment if no section

    #single-item lists to direct values
    for key in result:
        if isinstance(result[key], list) and len(result[key]) == 1:
            result[key] = result[key][0]

    return dict(result)

def transform_field(field_value):
    # Remove text between square brackets
    field_value = re.sub(r'\[.*?\]', '', field_value)
    # Split by ';' and strip whitespace
    return [item.strip() for item in field_value.split(';') if item.strip()]

def classify_application(x):
    data = []
    for classif in x.split(';'):
        data.append({'EP-CPCI' : {'section' : classif[0],
                                    'class': classif[1:3],
                                    'subclass': classif[3],
                                    'main group': classif.split('/')[0][4:],
                                    'subgroup': classif.split('/')[1] if '/' in classif else None}})
    return data

df_conversion = pd.read_csv('/scratch/students/ndillenb/metadata/processing/schema.csv', sep=';')
txt_to_mongodb_dict = dict(zip(df_conversion['txt'], df_conversion['mongodb']))

path = '/lhstdata1/patentdata/raw/gb/metadata'
file_count = sum(len(files) for _, _, files in os.walk(path))  
print('Total files:', file_count)
print('start processing...')
with tqdm(total=file_count) as pbar:
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root,file), "r", encoding='latin-1') as f:
                    data = {}
                    data['File_path']=os.path.join(root,file)
                    key = ''
                    value = ''
                    for line in f:
                        if line.startswith('#'):
                            if key!='':
                                data[key] = value[:-1] if value.endswith(' ') else value
                            value = ''
                            key = line.replace(':','').replace('#', '').replace('\n', '')
                        else:
                            value += line.replace('\n', ' ')
                    OPS_FD = data.get('OPS Family Data', None)
                    if isinstance(OPS_FD,str):
                        data['OPS Family Data'] = parse_patent_string(OPS_FD)     
                    data = {txt_to_mongodb_dict.get(k, k): v for k, v in data.items()}
                    data['Country'] = data['Publication Number'][:2] if 'Publication Number' in data else None
                    data['Kind'] = data['Publication Number'][-1] if 'Publication Number' in data else None
                    data['Inventor'] = transform_field(data['Inventor']) if 'Inventor' in data else None
                    data['Applicant'] = transform_field(data['Applicant']) if 'Applicant' in data else None
                    data['Classifiction'] = classify_application(data['IPC (International Patent Classification)']) if 'IPC (International Patent Classification)' in data else None
                    data['ID'] = data['Publication Number']
                    data['C_Application Date'] = data['Application Date'] if 'Application Date' in data else None
                    data['C_Publication Date'] = data['Publication Date'] if 'Publication Date' in data else None
                    collection.insert_one(data)
            pbar.update(1)