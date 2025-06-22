import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
import spacy
import pickle

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Custom Dataset class for text classification
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
   
    def __len__(self):
        return len(self.texts)
   
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.FloatTensor(self.labels[idx])
       
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
       
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# PyTorch model for multi-label classification with multi-layer classifier
class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(MultiLabelClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # For DistilBERT, use the first token's hidden state as the pooled output
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.classifier(output)

class Predictor_Class:
    def __init__(self):
        with open('mlb.pkl', 'rb') as f:
            mlb = pickle.load(f)
        model_name = 'AI-Growth-Lab/PatentSBERTa'  # Using DistilBERT as it's lighter than PatentSBERTa
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_classes = mlb.classes_.shape[0]  # Number of classes from MultiLabelBinarizer

        model = MultiLabelClassifier(model_name, num_classes).to(device)
        model.load_state_dict(torch.load('/scratch/students/ndillenb/metadata/classification/linear_best_model.pth', map_location=device))  # Load the best model if available
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.mlb = mlb

    def predict(self, text, threshold=0.7):
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            predictions = probabilities > threshold
        
        predicted_classes = self.mlb.classes_[predictions]
        predicted_probs = probabilities[predictions]
        
        return predicted_classes, predicted_probs