import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import os
import numpy as np

class MentalHealthClassifier:
    def __init__(self, model_dir='app/mental_health_model', tokenizer_dir='app/mental_health_tokenizer', label_encoder_path='label_encoder_classes.pkl'):
        print(f"Initializing MentalHealthClassifier...")
        print(f"Model directory: {model_dir}")
        print(f"Tokenizer directory: {tokenizer_dir}")
        print(f"Label encoder path: {label_encoder_path}")

        # Load the label encoder
        full_label_encoder_path = os.path.join(model_dir, label_encoder_path)
        print(f"Full label encoder path: {full_label_encoder_path}")
        try:
            with open(full_label_encoder_path, 'rb') as f:
                self.le_classes_ = pickle.load(f)
            print(f"Loaded label encoder classes: {self.le_classes_}")
            print(f"Number of classes in label encoder: {len(self.le_classes_)}")
        except Exception as e:
            print(f"Error loading label encoder: {str(e)}")
            self.le_classes_ = []

        # Load the tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            raise

        # Load the model
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = BertForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
            print(f"Number of classes in the model: {self.model.num_labels}")
            
            if len(self.le_classes_) != self.model.num_labels:
                print(f"Warning: Mismatch between number of classes in label encoder ({len(self.le_classes_)}) and model ({self.model.num_labels})")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        print("MentalHealthClassifier initialization completed.")

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        likelihood = probabilities[0][predicted_class].item()

        print(f"Raw model output: {outputs.logits}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted class: {predicted_class}")
        print(f"Number of classes in label encoder: {len(self.le_classes_)}")

        try:
            predicted_topic = self.le_classes_[predicted_class]
        except IndexError:
            print(f"Encountered unseen label: {predicted_class}")
            predicted_topic = "Unknown"

        return {
            "predicted_topic": predicted_topic,
            "likelihood": round(likelihood, 4),
            "predicted_class": predicted_class,
            "all_probabilities": probabilities[0].tolist()
        }

    def print_classes(self):
        for i, class_name in enumerate(self.le_classes_):
            print(f"Class {i}: {class_name}")