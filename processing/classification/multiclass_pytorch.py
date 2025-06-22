
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
import spacy
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MongoDB connection
client = MongoClient("localhost", 29012)
db = client["test-database"]
collection = db['aggregated-data']

#We load spacy for stopwords 
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

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

#model for multi-label classification
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

# Training function
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Apply sigmoid and threshold for multi-label prediction
            predictions = torch.sigmoid(outputs) > 0.5
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, f1, all_predictions, all_labels

"""
Load the data from MongoDB, preprocess it, and prepare it for training.
"""
# Data preprocessing
titles = []
classes = []
all_items = list(collection.find({'Country':'GB'}, {"Title": 1, "Classification":1}))

for item in all_items:
    if item['Title'] is not None and item['Title'] != '' and isinstance(item['Classification'], list) and len(item['Classification']) > 0:
        tmp_classes = []
        for element in item['Classification']:
            if isinstance(element, dict) and element is not None:
                for key in element.keys():
                    if key.startswith('EP-CPCI'):
                        tmp_classes.append(element[key]['section'] + element[key]['class'] + element[key]['subclass'])
        if len(tmp_classes) > 0:
            titles.append(item['Title'])
            classes.append(list(set(tmp_classes)))

# Multi-label binarization
mlb = MultiLabelBinarizer()
classes_bin = mlb.fit_transform(classes)

# Filter classes with at least 8 samples
# This is due to the fact that some classes may not have enough samples for training
# this will cause the unability to predict such class :/
class_counts = classes_bin.sum(axis=0)
valid_class_indices = np.where(class_counts >= 8)[0]
classes_bin = classes_bin[:, valid_class_indices]
mlb.classes_ = mlb.classes_[valid_class_indices]
# Save mlb for later use


with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)


count = 0
for i in range(len(classes_bin)):
    if not any(classes_bin[i]):
        count += 1
print(f"Number of items with no valid classes: {count}")

# Filter out samples with no valid classes
filtered_titles = [titles[i] for i in range(len(titles)) if any(classes_bin[i])]
filtered_classes_bin = classes_bin[np.any(classes_bin, axis=1)]

print(f"Total samples after filtering: {len(filtered_titles)}")
print(f"Number of classes: {filtered_classes_bin.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    filtered_titles, filtered_classes_bin, test_size=0.3, random_state=42
)

# Initialize tokenizer and model
model_name = 'AI-Growth-Lab/PatentSBERTa'  # Using PatentSBERTa
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_classes = y_train.shape[1]

model = MultiLabelClassifier(model_name, num_classes).to(device)

# Create datasets and data loaders
train_dataset = TextClassificationDataset(X_train, y_train, tokenizer)
test_dataset = TextClassificationDataset(X_test, y_test, tokenizer)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training setup
criterion = nn.BCEWithLogitsLoss()  # Good for multi-label classification
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Training loop
num_epochs = 5
best_f1 = 0


print("Starting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # Evaluate
    test_loss, test_f1, predictions, true_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Save best model
    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save(model.state_dict(), 'linear_best_model.pth')
        print(f"New best F1 score: {best_f1:.4f} - Model saved!")

# Load best model for final evaluation
model.load_state_dict(torch.load('linear_best_model.pth')) #Saving model
final_loss, final_f1, final_predictions, final_labels = evaluate(model, test_loader, criterion, device)

print(f"\nFinal Test Results:")
print(f"Loss: {final_loss:.4f}")
print(f"F1 Score: {final_f1:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(final_labels, final_predictions, target_names=mlb.classes_, zero_division=0))


"""
Now we evaluate the model.
"""

# Function to predict on new text
def predict_text(model, tokenizer, text, mlb, device, threshold=0.5):
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        predictions = probabilities > threshold
    
    predicted_classes = mlb.classes_[predictions]
    predicted_probs = probabilities[predictions]
    
    return predicted_classes, predicted_probs

# Eval prediction
if len(X_test) > 0:
    sample_text = X_test[0]
    predicted_classes, predicted_probs = predict_text(model, tokenizer, sample_text, mlb, device)
    
    print(f"\nExample prediction:")
    print(f"Text: {sample_text[:100]}...")
    print(f"Predicted classes: {predicted_classes}")
    print(f"Probabilities: {predicted_probs}")
    print(f"True classes: {mlb.classes_[y_test[0].astype(bool)]}")

