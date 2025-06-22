#%%
from pymongo import MongoClient
import os
import tempfile
import subprocess
import tarfile
from tqdm import tqdm
import time
from pymongo import UpdateOne
import sys
log = False
def create_queue(collection, size=100, start=0):
    pipeline = [
    {"$match": {"type": "text"}},  # Filter documents where 'type' is 'text'
    {"$match": {"OCR": {"$exists": False}}},  # Filter documents where 'OCR' field does not exist
    {"$project": {"_id": 1, "path":1, "page": 1, "Publication Number":1, "priority":1}},  # Extract only the path (_id) and pages fields
    {"$group": {"_id": "$Publication Number", "pages": {"$push": "$page"}, "paths": {"$push": "$path"}, "priority": {"$first": "$priority"}}},  # Group by 'path' and concatenate 'page'
    {"$sort": {"priority": -1}},  # Sort by 'priority' in descending order
    {"$skip": start},  # Skip the first 'start' documents
    {"$limit": size}  # Limit the result to 'size' documents
    #{"$sample": {"size": size}}  # Randomly select 10 documents
    ]
    return list(collection.aggregate(pipeline))

def extract_text(path, last_page):
    if path.endswith('.pdf'):
        return extract_images_from_pdf(path, last_page)
    elif ".tar.gz" in path:
        path = '/'.join(path.split('/')[:-1])
        return extract_tif_from_tar_gz(path, last_page)
    else:
        return []

def extract_images_from_pdf(pdf_path, end_page=None):
    extracted_files = []
    extracted_text = []
    print("Extracting images from PDF: ", pdf_path) if log else None
    with tempfile.TemporaryDirectory() as path:
        output = path+"/image"
        subprocess.run(["pdfimages", "-tiff" ,"-f", "1", "-l", str(end_page+1), pdf_path, output], check=True)
        extracted_files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.startswith("image")
        ])
        for i in range(len(extracted_files)):
            extracted_text.append(run_tesseract(extracted_files[i]))
    print("Extracted text from PDF: ", pdf_path) if log else None
    return extracted_text

def run_tesseract(image_path):
    print("Running tesseract on: ", image_path) if log else None
    output = image_path.replace(".tif", "")
    subprocess.run(["tesseract", image_path, output, "-l", "eng"])
    print("Tesseract finished on: ", image_path) if log else None
    with open(f"{output}.txt", "r") as file:
        return ' '.join(file.read().strip().split())

def extract_tif_from_tar_gz(tar_gz_path, end_page=None):
    extracted_files = []
    extracted_text = []
    print("Extracting tar.gz file: ", tar_gz_path) if log else None
    with tempfile.TemporaryDirectory() as path:
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".tif"):
                    extracted_files.append(os.path.join(path, member.name))
                    tar.extract(member, path=path)
        extracted_files = sorted(extracted_files)[:end_page+1]

        for i in range(len(extracted_files)):
            extracted_text.append(run_tesseract(extracted_files[i]))
    print("Extracted text from tar.gz file: ", tar_gz_path) if log else None
    return extracted_text


def __main__():
    client = MongoClient("localhost", 29012)
    db = client["test-database"]
    collection_CCN = db["CNN_GB"] #Database where to store the OCR results and locate the files to OCR
    queue = create_queue(collection_CCN, int(sys.argv[1].split('-')[0]), int(sys.argv[1].split('-')[1]))
    for item in tqdm(queue):
        path = item['paths'][0]
        print("starting to extract text for: ", path) if log else None
        extracted_text_value = extract_text(path, max(item['pages']))
        print("extracted text for: ", path) if log else None
        updates = []
        for i, text in enumerate(extracted_text_value):
            updates.append(UpdateOne({"Publication Number": item['_id'], "page": i}, {"$set": {"OCR": text}}))
        collection_CCN.bulk_write(updates)
__main__()