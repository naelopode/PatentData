from pymongo import MongoClient, InsertOne, UpdateOne
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from bson import ObjectId
import yaml
from rapidfuzz import process, fuzz
import rapidfuzz
from datasketch import MinHash, MinHashLSH
import re
from tqdm import tqdm
from collections import deque
from functools import lru_cache
# Before running, please download online list of business suffix
# !wget "https://raw.githubusercontent.com/ProfoundNetworks/company_designator/refs/heads/master/company_designator.yml"


def main():
    print("=" * 60)
    print("STARTING KEY PEOPLE PIPELINE")
    print("=" * 60)
    
    # Database connection
    client = MongoClient("localhost", 29012)
    db = client["test-database"]
    collection = db["aggregated-data4"] #Collection of patents
    collection_key_people = db["key_people4"] #Collection of entity nammed
    cluster_collection = db["cluster_keypeople4"] #Collection of clustered entity nammed
    
    print("\nSTEP 1: Loading Configuration")
    print("-" * 40)
    
    # Load company designator configuration
    with open('company_designator.yml', 'r') as file:
        try:
            company_designator = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error in configuration file: {exc}")
            return
    
    # Extract company abbreviations
    keys = list(company_designator.keys())
    list_company_abbr = []
    for key in keys:
        if company_designator[key]['lang'] in ['en', 'fr', 'de']:
            list_company_abbr.append(key)
            for item in company_designator[key]['abbr'] if 'abbr' in company_designator[key] else []:
                list_company_abbr.append(item)
    
    print(f"Loaded {len(list_company_abbr)} company abbreviations")
    
    # Define company titles and substitutions
    company_title = ['ag', 'a. g', 'c°', 'firm', 'akt.-ges', 'aktien-gesellschaft', 'ltd', 'co', 'limited', 
                    'works', 'company', 'corp', 'corporation', 'aktiengesellschaft', 'actiengesellschaft', 
                    'actien-gesellschaft', 'act.-ges', 'm.b.h', 'mbg', 'm. b. h', 'inc', 'ab', 'g.m.b.h', 
                    'gmbh', 'g. m. b. h', 'incorporated', 'inc', 'dev', 'cie']
    company_title = company_title + list_company_abbr
    
    substitutions = {
        r'\bgen\b': 'general',
        r'\bges\b': 'gesellschaft',
        r'\bbmfg\b': 'manufacturing',
        r'\bprod\b': 'productions',
        r'\bproduction\b': 'productions',
        r'\bmanufacturing\b': 'mfg',
        r'\bnat\b': 'national',
        r'\bmach\b': 'machine',
        r'\bint\b': 'international',
        r'\bind\b': 'industrial',
    }
    
    unwanted_chars = r'[,.;:\()\[\]{}\'\"`´]'
    replace_chars = r'[\s\.\-]'
    
    @lru_cache(maxsize=100000)
    def clean_names(name):
        # Remove company titles
        for title in company_title:
            name = re.sub(r'\b' + re.escape(title.lower()) + r'\b', '', name).strip()
        
        # Apply substitutions
        for pattern, replacement in substitutions.items():
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        
        # Remove content in brackets
        name = re.sub(r'\[.*?\]', '', name)
        
        # Remove unwanted characters
        name = re.sub(unwanted_chars, '', name)
        
        # Replace unwanted characters with space
        name = re.sub(replace_chars, ' ', name)
        
        # Remove multiple spaces
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def preprocess_name(name):
        tokens = sorted(name.split())  # Tokenization and sorting
        return " ".join(tokens)
    
    def text_cluster_applicants(list_applicants, list_ids):
        to_link_id = []
        to_link = []
        for i, applicant1 in enumerate(list_applicants):
            applicant1_clean = clean_names(applicant1.lower())
            for j, applicant2 in enumerate(list_applicants):
                if i != j and applicant1 != applicant2:
                    applicant2_clean = clean_names(applicant2.lower())
                    if (rapidfuzz.fuzz.ratio(applicant1_clean, applicant2_clean) > 60 or 
                        rapidfuzz.fuzz.token_ratio(applicant1_clean, applicant2_clean) > 67):
                        to_link_id.append(set([list_ids[i], list_ids[j]]))
                        to_link.append(set([applicant1, applicant2]))
        return to_link, to_link_id
    
    def update_tags(to_link_id):
        to_update = []
        for (id1, id2) in to_link_id:
            to_update.append(UpdateOne({'_id': id1}, {'$addToSet': {'matches': str(id2)}}))
            to_update.append(UpdateOne({'_id': id2}, {'$addToSet': {'matches': str(id1)}}))
        return to_update
    
    print("\nSTEP 2: Processing Key People Data")
    print("-" * 40)
    
    # Get all key people data
    applicants_data = collection_key_people.find({}, {'_id': 1, 'key_person': 1})
    df_applicant = pd.DataFrame(list(applicants_data))
    print(f"Found {len(df_applicant)} key people records")
    
    # Standardize names
    df_applicant["Standardized"] = df_applicant["key_person"].str.lower()
    print("Standardizing names...")
    df_applicant["Standardized"] = df_applicant["Standardized"].apply(clean_names)
    
    print("\nSTEP 3: Building LSH Index for Similarity Matching")
    print("-" * 40)
    
    # Initialize LSH
    lsh = MinHashLSH(threshold=0.75, num_perm=128)
    
    # Mappings
    name_to_id = {}
    id_to_name = {}
    
    # Add records to LSH
    print("Adding records to LSH index...")
    pbar = tqdm(total=len(df_applicant), desc="Building LSH index")
    for _, record in df_applicant.iterrows():
        pid = record["_id"]
        pname = preprocess_name(record["Standardized"])
        id_to_name[pid] = pname
        name_to_id[pname] = pid
        
        minhash = MinHash(num_perm=128)
        for token in pname.split():
            minhash.update(token.encode("utf8"))
        lsh.insert(pid, minhash)
        pbar.update(1)
    pbar.close()
    
    # Find matches
    print("Finding similar records...")
    updates = []
    pbar = tqdm(total=len(df_applicant), desc="Finding matches")
    for _, record in df_applicant.iterrows():
        pid = record["_id"]
        pname = preprocess_name(record["Standardized"])
        
        minhash = MinHash(num_perm=128)
        for token in pname.split():
            minhash.update(token.encode("utf8"))
        
        similar_ids = lsh.query(minhash)
        similar_ids = [str(sid) for sid in similar_ids if sid != pid]
        
        updates.append(UpdateOne({"_id": pid}, {"$set": {"matches": similar_ids}}))
        pbar.update(1)
    pbar.close()
    
    # Apply updates
    if updates:
        print(f"Applying {len(updates)} match updates...")
        collection_key_people.bulk_write(updates)
    
    print("\nSTEP 4: Adding Cross-Application Matches")
    print("-" * 40)
    
    # Build key person to ID mapping
    keypeople2id = collection_key_people.find({}, {'_id': 1, 'key_person': 1})
    keypeople2id = {item['key_person']: item['_id'] for item in keypeople2id}
    
    updates = []
    bulk_size = 10000
    
    for item in tqdm(collection.find({}, {'_id': 1, 'key_people': 1}), desc="Processing applications"):
        if 'key_people' in item and item['key_people']:
            ids = []
            for people in item['key_people']:
                if people in keypeople2id:
                    ids.append(keypeople2id[people])
            
            if len(ids) > 1:
                to_link, to_link_id = text_cluster_applicants(item['key_people'], ids)
                if len(to_link_id) > 0:
                    to_update = update_tags(to_link_id)
                    updates.extend(to_update)
        
        if len(updates) > bulk_size:
            collection_key_people.bulk_write(updates)
            updates = []
    
    if updates:
        collection_key_people.bulk_write(updates)
    
    print("\nSTEP 5: Adding Name Matches")
    print("-" * 40)
    
    # Create ID to key person mapping
    id2keyperson = {str(item['_id']): item['key_person'] 
                   for item in collection_key_people.find({}, {'_id': 1, 'key_person': 1})}
    
    updates = []
    for item in tqdm(collection_key_people.find({}, {'_id': 1, 'matches': 1}), desc="Adding name matches"):
        if 'matches' in item and len(item['matches']) > 0:
            names_matches = []
            for match in item['matches']:
                if match in id2keyperson:
                    names_matches.append(id2keyperson[match])
            
            if names_matches:
                updates.append(UpdateOne({"_id": item["_id"]}, {"$set": {"names_matches": names_matches}}))
        
        if len(updates) > 500:
            collection_key_people.bulk_write(updates)
            updates = []
    
    if updates:
        collection_key_people.bulk_write(updates)
    
    print("\nSTEP 6: Creating Clusters")
    print("-" * 40)
    
    # Clear existing clusters
    cluster_collection.delete_many({})
    cluster_collection.create_index([("_id", 1)])
    cluster_collection.create_index([("matches", 1)])
    
    # Build adjacency list
    print("Building adjacency list...")
    adj_list = {}
    cursor = collection_key_people.find({}, {"_id": 1, "matches": 1})
    
    for doc in cursor:
        node_id = ObjectId(doc["_id"])
        if node_id not in adj_list:
            adj_list[node_id] = set()
        
        for match in doc.get("matches", []):
            match_id = ObjectId(match)
            adj_list[node_id].add(match_id)
            if match_id not in adj_list:
                adj_list[match_id] = set()
            adj_list[match_id].add(node_id)
    
    # Find connected components using BFS
    print("Finding connected components...")
    visited = set()
    clusters = []
    
    def bfs(start_node):
        queue = deque([start_node])
        cluster = set()
        
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            cluster.add(node)
            for neighbor in adj_list.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return cluster
    
    # Identify clusters
    for node in adj_list:
        if node not in visited:
            cluster = bfs(node)
            clusters.append(cluster)
    
    print(f"Found {len(clusters)} clusters")
    
    # Build structured cluster data
    print("Building cluster data...")
    bulk_insert = []
    
    for cluster in tqdm(clusters, desc="Processing clusters"):
        cluster_elements = list(cluster)
        
        # Fetch names for cluster elements
        cursor = collection_key_people.find({"_id": {"$in": cluster_elements}}, {"_id": 1, "key_person": 1})
        elements_data = [{"id": doc["_id"], "key_person": doc["key_person"]} for doc in cursor]
        
        # Use first element's name as representative
        main_name = elements_data[0]["key_person"] if elements_data else None
        alias = [element["key_person"] for element in elements_data]
        
        bulk_insert.append({
            "main_person": main_name,
            "elements": elements_data,
            "alias": alias
        })
    
    # Save clusters
    if bulk_insert:
        cluster_collection.insert_many(bulk_insert)
    
    print("\nSTEP 7: Assigning Clusters to Applications")
    print("-" * 40)
    
    # Load cluster data into dictionary
    cluster_map = {}
    for doc in cluster_collection.find({}, {'alias': 1, 'main_person': 1}):
        if isinstance(doc['alias'], list):
            for alias in doc['alias']:
                cluster_map[alias] = {'main_person': doc['main_person'], '_id': str(doc['_id'])}
        else:
            cluster_map[doc['alias']] = {'main_person': doc['main_person'], '_id': str(doc['_id'])}
    
    # Update applications with cluster information
    updates = []
    batch_size = 10000 #Select batch size for bulk updates
    
    for item in tqdm(collection.find({'key_people': {'$not': {'$eq': None}}}, {'_id': 1, 'key_people': 1}), 
                    desc="Assigning clusters"):
        applicant_metadata = {'alias': [], 'ids': [], 'names': []}
        
        for element in item["key_people"]:
            if element in cluster_map:
                cluster_id = cluster_map[element]['_id']
                if cluster_id not in applicant_metadata['ids']:
                    applicant_metadata['names'].append(cluster_map[element]['main_person'])
                    applicant_metadata['ids'].append(cluster_id)
        
        applicant_metadata['alias'] = applicant_metadata['names']
        
        # Assign unique ID if all belong to same cluster
        applicant_metadata['id'] = (applicant_metadata['ids'][0] 
                                  if len(set(applicant_metadata['ids'])) == 1 
                                  else None)
        
        updates.append(UpdateOne({"_id": ObjectId(item["_id"])}, 
                                {"$set": {"cluster_keypeople": applicant_metadata}}))
        
        if len(updates) >= batch_size:
            collection.bulk_write(updates)
            updates = []
    
    # Final batch
    if updates:
        collection.bulk_write(updates)
    
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Processed {len(df_applicant)} key people records")
    print(f"Created {len(clusters)} clusters")
    print(f"Updated application records with cluster information")
    print("=" * 60)


if __name__ == "__main__":
    main()