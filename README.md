# Mental Health Problem Classifier API

## Introduction

Mental health issues are a growing concern worldwide, with an increasing number of individuals experiencing conditions like anxiety, depression, and stress. Early detection and intervention are crucial for providing support and treatment. This project presents a **Mental Health Problem Classifier API** built using FastAPI and a BERT-based machine learning model. The API allows for the classification of mental health problems based on textual input, enabling developers and healthcare professionals to integrate this tool into applications for monitoring and supporting mental well-being.

## Methods and Results

### Approach

The classifier leverages a pre-trained BERT model fine-tuned on a dataset of counseling conversations to understand and categorize text inputs into specific mental health issues. The steps involved in building the model include:

1. **Data Collection**: Utilizing the [CounselChat](https://counselchat.com/) dataset containing real-life counseling questions and responses.
2. **Data Preprocessing**: Cleaning the text data, handling missing values, and preparing the input for the BERT tokenizer.
3. **Model Training**:
   - Employed a pre-trained BERT model from Hugging Face Transformers library.
   - Fine-tuned the model on the prepared dataset for multi-class classification over 10 epochs.
   - Utilized techniques like learning rate scheduling and early stopping to optimize performance.
4. **Evaluation**: Assessed the model using metrics such as training loss, validation loss, and validation accuracy at each epoch.

### Results

The fine-tuned BERT model was trained over 10 epochs, and the training progress is summarized below:

| **Epoch** | **Average Training Loss** | **Validation Loss** | **Validation Accuracy** |
| --------- | ------------------------- | ------------------- | ----------------------- |
| 1         | 2.7834                    | 2.1490              | 46.01%                  |
| 2         | 1.8333                    | 1.5469              | 59.32%                  |
| 3         | 1.2829                    | 1.1495              | 69.48%                  |
| 4         | 0.9064                    | 0.9001              | 74.63%                  |
| 5         | 0.6710                    | 0.7918              | 77.93%                  |
| 6         | 0.5208                    | 0.7485              | 78.64%                  |
| 7         | 0.4158                    | 0.7450              | 79.11%                  |
| 8         | 0.3416                    | 0.7404              | 79.11%                  |
| 9         | 0.2831                    | 0.7582              | 79.58%                  |
| 10        | 0.1992                    | 0.7585              | 79.58%                  |

_Note: The above statistics are derived from the training logs._

By the end of training, the model achieved:

- **Average Training Loss**: **0.1992**
- **Validation Loss**: **0.7585**
- **Validation Accuracy**: **79.58%**

These results indicate that the model effectively understands and categorizes mental health-related text, making it a valuable tool for practitioners and developers.

## Discussion

Developing the Mental Health Problem Classifier presented several challenges:

- **Data Imbalance**: Some mental health categories had significantly more data than others. Techniques like oversampling and class weighting were employed to address this.
- **Sensitive Nature of Data**: Ensuring the ethical use of sensitive data required careful consideration of privacy and compliance with data protection regulations.
- **Model Interpretability**: Explaining the model's decisions is crucial in healthcare applications. Future work includes integrating explainable AI techniques to provide insights into the model's predictions.

**Lessons Learned**:

- The importance of data quality and preprocessing in building robust NLP models.
- The need for continuous evaluation and updating of the model to adapt to evolving language use around mental health.

**Future Work**:

- Incorporate additional data sources to enhance model robustness.
- Deploy the API in a real-world application for user testing and gather feedback.
- Explore multilingual support to cater to non-English speaking populations.
- Implement explainable AI methods to improve transparency and trust in model predictions.

## API Usage

### Predict Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:

  ```json
  {
    "text": "My patient has been feeling really anxious lately and can't sleep."
  }
  ```

- **Response**:

  ```json
  {
    "predicted_topic": "Anxiety",
    "likelihood": 0.7958
  }
  ```

### Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Response**: Welcome message and basic API information.

## Optional Interactive Elements

A live demo of the API is deployed at Azure and accessible at [https://mental-health-classifier.azurewebsites.net/docs](https://mental-health-classifier.azurewebsites.net/docs)

## Project Structure

```markdown:Project Structure
therapy-chat-fastapi/
├── app/
│   ├── mental_health_model/
│   │   ├── config.json
│   │   ├── label_encoder_classes.pkl
│   │   ├── model.safetensors
│   │   ├── num_classes.txt
│   │   └── pytorch_model.bin
│   ├── mental_health_tokenizer/
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── __init__.py
│   ├── main.py
│   └── model.py
├── data/
│   └── 20200325_counsel_chat.csv
├── .dockerignore
├── .env
├── .gitattributes
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── train_model.py
```

## Getting Started

### Prerequisites

- **Docker**: For containerized deployment.
- **Python 3.12**: For local development and testing.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/therapy-chat-fastapi.git
   cd therapy-chat-fastapi
   ```

2. **Build the Docker image**:

   ```bash
   docker build -t mental-health-classifier .
   ```

3. **Run the Docker container**:

   ```bash
   docker run -p 8000:8000 mental-health-classifier
   ```

4. **Access the API**:

   Navigate to [http://localhost:8000](http://localhost:8000) or [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API documentation.

## Local Development

For development and testing without Docker:

1. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server**:

   ```bash
   uvicorn app.main:app --reload
   ```

## Training the Model

To retrain the model with new data:

1. **Update the dataset**:

   Replace or update the `data/20200325_counsel_chat.csv` file with new training data.

2. **Run the training script**:

   ```bash
   python train_model.py
   ```

   This script will preprocess the data, fine-tune the BERT model, and save the updated model files in the `app/mental_health_model/` directory.

### Training Logs

An excerpt of the training logs is shown below:

```training_logs.txt
Epoch 1/10
Average training loss: 2.7834
Validation Loss: 2.1490, Validation Accuracy: 46.01%

Epoch 2/10
Average training loss: 1.8333
Validation Loss: 1.5469, Validation Accuracy: 59.32%

Epoch 3/10
Average training loss: 1.2829
Validation Loss: 1.1495, Validation Accuracy: 69.48%

Epoch 4/10
Average training loss: 0.9064
Validation Loss: 0.9001, Validation Accuracy: 74.65%

Epoch 5/10
Average training loss: 0.6710
Validation Loss: 0.7918, Validation Accuracy: 77.93%

Epoch 6/10
Average training loss: 0.5208
Validation Loss: 0.7485, Validation Accuracy: 78.64%

Epoch 7/10
Average training loss: 0.4158
Validation Loss: 0.7450, Validation Accuracy: 79.11%

Epoch 8/10
Average training loss: 0.3416
Validation Loss: 0.7404, Validation Accuracy: 79.11%

Epoch 9/10
Average training loss: 0.2831
Validation Loss: 0.7582, Validation Accuracy: 79.58%

Epoch 10/10
Average training loss: 0.1992
Validation Loss: 0.7585, Validation Accuracy: 79.58%
```
