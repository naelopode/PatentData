import os
from pymongo import MongoClient
import subprocess
import time
def create_queue(collection): #This check how many items we should do
    pipeline = [
    {"$match": {"type": "text"}},  # Filter documents where 'type' is 'text'
    {"$match": {"OCR": {"$exists": False}}},  # Filter documents where 'OCR' field does not exist
    {"$project": {"_id": 1, "path":1, "page": 1, "Publication Number":1}},  # Extract only the path (_id) and pages fields
    {"$group": {"_id": "$Publication Number", "pages": {"$push": "$page"}, "paths": {"$push": "$path"}, "priority": {"$first": "$priority"}}},  # Group by 'path' and concatenate 'page'
    {"$sort": {"priority": -1}}  # Sort by 'priority' in descending order
    ]
    return len(list(collection.aggregate(pipeline)))
def define_args(nb_sessions):
    client = MongoClient("localhost", 29012)
    db = client["test-database"]
    collection_CCN = db["CNN_GB"] 
    size_db = create_queue(collection_CCN)
    if size_db > 0:
        per_sessions = size_db // nb_sessions
        remainder = size_db % nb_sessions
        args = []
        for i in range(nb_sessions):
            if i == nb_sessions - 1:
                args.append(f"{per_sessions+remainder}-{per_sessions*i+1}")
            else:
                args.append(f"{per_sessions}-{per_sessions*i}")
        return args
    else:
        return None
def __main__():
    # Define the number of sessions
    while True:
        nb_sessions = 20 # Definie the number of parallel sessions
        print("Launching parallel OCR sessions...")
        # Call the function to define arguments
        args = define_args(nb_sessions)
        if args is None:
            print("Waiting for more data, currently")
            return
        else:
            args = " ".join(args)
            print(f"With Args: {args}")
            command = ' '.join(["parallel", "/scratch/students/ndillenb/OCR/ocr_env/bin/python", "/scratch/students/ndillenb/OCR/tesseract_pipeline.py", ":::", args])
            subprocess.run(command, shell=True, executable='/bin/bash', check=True)
            print("Parallel OCR sessions launched successfully.")
        time.sleep(60)
__main__()
