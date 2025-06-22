import argparse
import pickle
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from tqdm import tqdm

def find_classification(classification):
    classifications = []
    filter_classifications = []
    for item in classification:
        for element in item.find_all('li'):
            classifications.append(element.find(attrs={"itemprop": "Code"}).get_text())
    for item in classifications[::-1]:
        if not any(element.startswith(item) for element in filter_classifications):      
            filter_classifications.append(item)
    return filter_classifications

def get_google_patent(ID):
    url = f'http://patents.google.com/patent/{ID}/en'
    proxies={
        "http": "socks5://LOGIN:PASSWORD@SERVICE:80/",
        "https": "socks5://LOGIN:PASSWORD@SERVICE:80/"
    }

    try:
        response = requests.get(url, proxies=proxies, timeout=10)
        response.raise_for_status()
    except NameError as err:
        print(f"HTTP error occurred: {err}")
        return None
    content = BeautifulSoup(response.content, 'html.parser')

    title = content.find('title').get_text().replace(ID, '').replace(' - Google Patents', '').replace('-', '').strip()
    ocr_element = content.find(class_='description')
    ocr = ocr_element.get_text() if ocr_element else ""
    pub_date = content.find(attrs={"itemprop": "publicationDate"}).get_text()
    app_date = content.find(attrs={"itemprop": "filingDate"}).get_text()
    classification = find_classification(content.find_all(attrs={"itemprop": "classifications"}))
    google_patent = {
        'Publication Number': ID,
        'Title': title,
        'ocr': ocr,
        'Publication Date': pub_date,
        'Application Date': app_date,
        'classification': classification
    }
    return google_patent

def main(argument):
    print(f"Received argument: {argument}")

    with open('/scratch/students/ndillenb/metadata/analysis/google-patents3.pickle', 'rb') as f:
        missing_pub = pickle.load(f)
    size, start = argument.split(':')
    missing_pub = missing_pub[int(start):int(start)+int(size)]
    client = MongoClient("localhost", 29012)
    db = client["test-database"]
    collection = db["collection-json"]

    item_id = list(collection.find({},{'_id':1, 'ID':1}))
    dict_ID2id = {item['ID']: item['_id'] for item in item_id}

    for item in tqdm(missing_pub):
        google_data = get_google_patent(item)
        collection.update_one({'_id': dict_ID2id[item]}, {'$set': {'google-patent':google_data}})

    # Your main logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape OCR data from Google Patents.")
    parser.add_argument("argument", type=str, help="START:FINISH for the range of patents to scrape")
    args = parser.parse_args()
    
    main(args.argument)