from pymongo import MongoClient
import os
import json
from tqdm import tqdm
import datetime

client = MongoClient("localhost", 29012)
db = client["test-database"]
collection = db["collection-json"]
COUNTRY = 'DE'
already_there = list(db["collection-txt2"].find({'Country':COUNTRY}, {'ID':1})) + list(db["collection-json"].find({'Country':COUNTRY}, {'ID':1}))
already_there_ids = [i['ID'] for i in already_there]

#get nested data safely
def get_nested(data, keys, default=None):
    """Retrieve a nested value from a dictionary, returning default if not found."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

def find_ealiest(dict_dates):
    dates = []
    for date_tmp in dict_dates:
        try:
            if dict_dates[date_tmp] != None:
                dates.append(datetime.strptime(dict_dates[date_tmp], '%Y-%m-%d'))
        except:
            pass
    if len(dates) == 0:
        return None
    return min(dates)

def get_nested_wl(data, keys, PatentID, default=None):
    """Retrieve a nested value from a dictionary, returning default if not found."""
    for key in keys:
        if key == 'exchange-documents':
            if isinstance(data, list):
                for item in data:
                    if get_nested(item, ['exchange-documents', 'exchange-document', '@country']) + get_nested(item,  ['exchange-documents', 'exchange-document','@doc-number']) + get_nested(item,  ['exchange-documents', 'exchange-document','@kind']) == PatentID:
                        data = item
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

#parse date safely with return None
def parse_date(date_str, date_format='%Y%m%d'):
    """Convert date string to 'YYYY-MM-DD' format, return None if invalid."""
    try:
        return datetime.datetime.strptime(date_str, date_format).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

#extract multiple priority claims
def extract_priority_claims(priority_data):
    """Extract multiple priority claims if available."""
    if not priority_data:
        return None
    claims = []
    priority_claims = priority_data if isinstance(priority_data, list) else [priority_data]
    for claim in priority_claims:
        claims.append({
            'Number': get_nested(claim, ['document-id', 'doc-number', '$']),
            'Date': parse_date(get_nested(claim, ['document-id', 'date', '$'])),
            'Kind': get_nested(claim, ['@kind'])
        })
    return claims

#extract multiple classification data
def extract_classifications(classification_list):
    """Extract all available classifications."""
    classifications = {}
    if isinstance(classification_list, dict):
        classification_list = [classification_list]
    for classification in classification_list:
        scheme = f"{classification.get('classification-scheme', {}).get('@office', 'unknown')}-"\
                 f"{classification.get('classification-scheme', {}).get('@scheme', 'unknown')}-"\
                 f"{classification.get('@sequence', 'unknown')}" 
        classifications[scheme] = {key: value['$'] for key, value in classification.items() if key not in ['@sequence', 'classification-scheme']}
    return classifications if classifications else None

#extract applicants
def extract_applicants(applicants_data):
    """Extract applicant names if available."""
    if not applicants_data:
        return None
    applicants = []
    applicants_list = applicants_data if isinstance(applicants_data, list) else [applicants_data]
    for applicant in applicants_list:
        applicants.append(get_nested(applicant, ['applicant-name', 'name', '$']))
    return applicants

#extract references
def extract_references(reference_data):
    """Extract references if available in citations."""
    if not reference_data:
        return None
    references = {}
    for ref in reference_data:
        ref_type = ref.get('@document-id-type', 'unknown')
        references[ref_type] = {key: value['$'] for key, value in ref.items() if key != '@document-id-type'}
    return references


path = f'/lhstdata1/patentdata/raw/{COUNTRY.lower()}/metadata/'
def get_keys(d, keys=[]):
    for k, v in d.items():
        keys.append(k)
        if isinstance(v, dict):
            get_keys(v, keys)
    return keys
file_count = sum(len(files) for _, _, files in os.walk(path))  
print('Total files:', file_count)
print('start processing...')

updates = []
with tqdm(total=file_count) as pbar:
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                if file.split('/')[-1].replace('.json', '').replace('.','') in already_there_ids:
                    print(f'Already imported {file}')
                    pbar.update(1)
                    continue
                with open(os.path.join(root, file), "r") as f:
                    data_org = json.load(f)
                #extract values
                ID = file.split('/')[-1].replace('.json', '').replace('.','')
                bibliographic_data = get_nested_wl(data_org, ['ops:world-patent-data', 'ops:equivalents-inquiry', 'ops:inquiry-result',
                                                        'exchange-documents', 'exchange-document', 'bibliographic-data'], ID,{})

                inventors_list = get_nested(bibliographic_data, ['parties', 'inventors', 'inventor'], [])
                if inventors_list and not isinstance(inventors_list, list):
                    inventors_list = [inventors_list]
                applications_list = get_nested(bibliographic_data, ['application-reference', 'document-id'], [])
                publication_date_list = get_nested(bibliographic_data, ['publication-reference', 'document-id'], [])
                classification_list = get_nested(bibliographic_data, ['patent-classifications', 'patent-classification'], [])
                references_list = get_nested(bibliographic_data, ['citations', 'citation'], [])
                priority_claims = get_nested(bibliographic_data, ['priority-claims', 'priority-claim'], [])
                abstracts_list = get_nested(bibliographic_data, ['abstract'], [])
                applicants_list = get_nested(bibliographic_data, ['parties', 'applicants', 'applicant'], [])
                #Construct
                data = {
                    'Database': get_nested(data_org, ['ops:world-patent-data', '@xmlns', 'ops']),
                    'File_path': os.path.join(root, file),
                    'Patmonitor Version': None,
                    'Download Date': None,
                    'Sub Database': None,
                    'Title': get_nested(bibliographic_data, ['invention-title', '$']),
                    'Publication Number': get_nested(data_org, ['ops:world-patent-data', 'ops:equivalents-inquiry', 'ops:publication-reference', 'document-id', 'doc-number', '$']),
                    'Country': get_nested(data_org, ['ops:world-patent-data', 'ops:equivalents-inquiry', 'ops:publication-reference', 'document-id', 'country', '$']),
                    'Doc_kind': get_nested(data_org, ['ops:world-patent-data', 'ops:equivalents-inquiry', 'ops:publication-reference', 'document-id', 'kind', '$']),
                    'Publication Date': {pub.get('@document-id-type', 'unknown'): parse_date(pub.get('date', {}).get('$'))
                                        for pub in publication_date_list},
                    'Inventor': [get_nested(inv, ['inventor-name', 'name', '$']).replace('\u2002',' ') for inv in inventors_list if isinstance(inv, dict)],
                    #'Inventor': {inv.get('@data-format', 'unknown'): get_nested(inv, ['inventor-name', 'name', '$'])
                    #            for inv in inventors_list if isinstance(inv, dict)},
                    'Applicant': extract_applicants(applicants_list) if isinstance(applicants_list, (list, dict)) else None,
                    'Requested Patent': None,
                    'Application Number': {app.get('@document-id-type', 'unknown'): get_nested(app, ['doc-number', '$'])
                                        for app in applications_list},
                    'Application Date': {app.get('@document-id-type', 'unknown'): parse_date(get_nested(app, ['date', '$']))
                                        for app in applications_list},
                    'Application Country': {app.get('@document-id-type', 'unknown'): get_nested(app, ['country', '$'])
                                            for app in applications_list},
                    'Priority': extract_priority_claims(priority_claims),
                    'Classification': [extract_classifications(classification_list)],
                    'IPC': None,
                    'ICP': None,
                    'CPC': None,
                    'NCL': None,
                    'MCS': None,
                    'MCA': None,
                    'Classification IPCR': get_nested(bibliographic_data, ['classifications-ipcr', 'classification-ipcr', 'text', '$']),
                    'Family ID': get_nested(data_org, ['ops:world-patent-data', 'ops:equivalents-inquiry', 'ops:inquiry-result',
                                                    'exchange-documents', 'exchange-document', '@family-id']),
                    'OPS Family Data': None,
                    'OPS Simple Family Data': None,
                    'Inpadoc Family ID': None,
                    'Abstract': get_nested(bibliographic_data, ['abstract', '$']),
                    'Abstract_EN': next((abs_data['$'] for abs_data in abstracts_list if abs_data.get('@lang') == 'en'), None),
                    'Priority Claims': extract_priority_claims(priority_claims),
                    'References': extract_references(references_list),
                    'raw': data_org
                }
                data['C_Publication Date'] = min((date for date in data['Publication Date'].values() if date is not None), default=None)
                data['ID']=data['Country']+data['Publication Number']+data['Doc_kind']
                data['C_Application Date'] = min((date for date in data['Application Date'].values() if date is not None), default=None)
                updates.append(data)
                #collection.insert_one(data)
                #print(json.dumps(data, indent=1))
            if len(updates) >= 1000:
                collection.insert_many(updates)
                updates = []
            pbar.update(1)
    if updates:
        collection.insert_many(updates)
        updates = []