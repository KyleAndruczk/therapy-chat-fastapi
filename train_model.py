import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import pickle

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    print("Starting Mental Health Problem Classifier training...")

    # Load the data
    print("Loading data...")
    df = pd.read_csv('data/20200325_counsel_chat.csv')
    print(f"Loaded {len(df)} records.")

    # Encode labels
    print("Encoding labels...")
    le = LabelEncoder()
    df['topic_encoded'] = le.fit_transform(df['topic'])
    print(f"Number of unique topics: {len(le.classes_)}")

    # Save the label encoder using pickle
    label_encoder_path = 'app/mental_health_model/label_encoder_classes.pkl'
    print(f"Saving label encoder classes to: {label_encoder_path}")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(le.classes_, f)
    print("Label encoder classes saved successfully.")

    # Split the data
    print("Splitting data into train and validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['questionText'].tolist(), 
        df['topic_encoded'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    print(f"Train set size: {len(train_texts)}, Validation set size: {len(val_texts)}")

    # Load pre-trained BERT tokenizer and model
    print("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))
    print("BERT model and tokenizer loaded successfully.")

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    print("Datasets and dataloaders created.")

    # Training setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        print("Running validation...")
        model.eval()
        val_accuracy = 0
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                val_accuracy += torch.sum(preds == labels).item()
        
        val_loss /= len(val_loader)
        val_accuracy /= len(val_dataset)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    print("Training completed.")

    # Save the number of classes
    num_classes_path = 'app/mental_health_model/num_classes.txt'
    print(f"Saving number of classes to: {num_classes_path}")
    with open(num_classes_path, 'w') as f:
        f.write(str(len(le.classes_)))
    print("Number of classes saved successfully.")

    # Save the model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained('app/mental_health_model')
    tokenizer.save_pretrained('app/mental_health_tokenizer')
    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    train_model()