![](images/banner.png)

This repository contains the code used in my Master's thesis at EPFL, LHST Lab (2025).
The tools developed here are designed to process and analyze patent data, focusing on the extraction of information from patent documents from images with OCR and NLP, classification of patents using CPC classification, and clustering inventors/assignee...
# PatentData Repository Structure
```
📦 PatentData
├─ analysis
│  ├ rename.json 
│  └ template.json # How to use the data for analysis
├─ clustering
│  └─ clustering.py # Patentee clustering algorithm
├─ processing
│  ├─ classification
│  │  └─ multiclass_pytorch.py # Use a multiclass multioutput classifier to classify patent titles in CPC Classifications
│  ├─ extractio
│  │  ├─ ID2Dates
│  │  │  └─ german_dates.ipynb # Validate hypothesis that you can predict publication dates based on patent ids.
│  │  ├─ llm
│  │  │  ├─ compare_json.ipynb # Compare the results of different extraction methods
│  │  │  ├─ gemma.ipynb
│  │  │  ├─ google_api.ipynb
│  │  │  ├─ LambdaLabda_api.ipynb
│  │  │  └─ OpenAI_api.ipynb
│  │  └─ spacy
│  │     ├─ run_spacy.py # Execute extraction on OCR
│  │     ├─ spacy_de.ipynb # Train german spacy model
│  │     ├─ spacy_fr.ipynb # Train french spacy model
│  │     ├─ spacy_gb.ipynb # Train british spacy model
│  │     └─ spacy.ipyb # Train american spacy model
│  ├─ ocr
│  │  ├─ page_labels # This part is for predicting patent page type
│  │  │  ├─ create_dataloader.py
│  │  │  ├─ create_model.py
│  │  │  └─ preditct_type.py
│  │  ├─ parellel_start.py
│  │  └─ tesseract_pipeline.py
│  ├─ scraper 
│  │  ├─ fetch_pdf_us.ipynb
│  │  └─ google_patents_scrape.py
│  └─ upload
│     ├─ aggregate.py
│     ├─ classification_predictor.py
│     ├─ import_json.py # Upload json patent metadata to the database
│     ├─ import_txt.py # Upload txt patent metdata to the database
│     ├─ schema.csv # Schema from json and txt to database
│     └─ upload_text.py # Upload already OCRized text to the database
├─ images
│  └─ banner.png
├─ README.md
├─ thesis.pdf
└─ requirements.txt
```
# Installation
To run the code in this repository, you need to install the required packages. You can do this by running the following command in your terminal:
```bash
pip install -r requirements.txt
```
The dataset is contained in MongoDB, you can run one in a docker container:
```bash
docker run --name mongodb -d -p 27017:27017 mongo
```

You then need to import the dataset into MongoDB. To load the .dump file, you can use the following command:
```bash
mongorestore --db patentdata --drop /path/to/patentdata.dump
```

The dataset is available in the `patentdata.dump` file, which can be found on [Kaggle](https://www.kaggle.com/datasets/naelopode/patentdata)

