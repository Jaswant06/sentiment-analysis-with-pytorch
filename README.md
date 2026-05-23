# Sentiment Analysis With PyTorch

## Project Goal

This project builds a sentiment analysis model using PyTorch.

The model classifies text as:

```text
positive
negative
```

The main goal is to learn deep learning for text data using:

- tokenization
- vocabulary building
- text-to-number conversion
- padding
- embeddings
- LSTM
- bidirectional LSTM
- binary classification
- training curves
- confusion matrix
- model saving and loading
- interactive prediction

## Dataset

This project uses a Twitter sentiment dataset.

The original dataset contains tweets labeled as:

```text
positive
negative
neutral
```

For this project, only the positive and negative rows were used. Neutral tweets were removed to create a binary classification problem.

Final dataset size used:

```text
16,363 tweets
```

Train/validation split:

```text
Training examples: 13,090
Validation examples: 3,273
```

## Tools Used

- Python
- PyTorch
- pandas
- scikit-learn
- matplotlib
- seaborn

## Model Architecture

The model uses a bidirectional LSTM.

Architecture:

```text
Input text
Tokenization
Vocabulary lookup
Padding to fixed length
Embedding layer
Bidirectional LSTM
Dropout
Linear layer
Binary sentiment output
```

Model details:

```text
Embedding size: 64
LSTM hidden size: 128
Bidirectional: True
Final linear input: 256
Output: 1 logit
```

The model uses `BCEWithLogitsLoss` because this is a binary classification task.

## Training Results

Best validation performance:

```text
Validation Accuracy: 86.40%
F1 Score: 87.03%
```

The first simple LSTM reached around 60% validation accuracy. After upgrading to a bidirectional LSTM, validation accuracy improved to around 86%.

## Outputs

The training script saves:

```text
models/sentiment_lstm.pth
models/vocab.json
plots/training_curves.png
plots/confusion_matrix.png
```

The prediction script loads the saved model and vocabulary, then lets the user type custom text and receive a sentiment prediction.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run predictions:

```bash
python src/predict.py
```

Example prediction:

```text
Input:
I love this movie, it was amazing

Output:
Prediction: positive
Confidence: 98.42%
```

## Project Structure

```text
sentiment-analysis-with-pytorch/
+-- data/
|   +-- train.csv
+-- models/
|   +-- sentiment_lstm.pth
|   +-- vocab.json
+-- plots/
|   +-- training_curves.png
|   +-- confusion_matrix.png
+-- src/
|   +-- train.py
|   +-- predict.py
+-- .gitignore
+-- README.md
+-- requirements.txt
```

## What I Learned

This project taught the full deep learning workflow for text:

- raw text cannot go directly into a neural network
- text must first be converted into tokens
- tokens are mapped to integer IDs using a vocabulary
- sequences must be padded to the same length
- embeddings convert word IDs into dense vectors
- LSTMs process text as a sequence
- bidirectional LSTMs read text in both directions
- binary classification can be done with one output neuron and sigmoid probability
- F1 score and confusion matrix give more detail than accuracy alone

## Limitations

This model only predicts:

```text
positive
negative
```

It does not understand neutral sentiment, mixed emotions, sarcasm, or multiple emotions at once.

A better next version would use:

```text
negative
neutral
positive
```

An even stronger future version would use multi-label emotion classification, where one sentence can have multiple emotions at the same time.

Example:

```text
joy: 0.72
sadness: 0.41
anger: 0.05
fear: 0.12
```

## Next Step

The next project is:

```text
Multi-Label Emotion Classification With PyTorch
```

That project will classify text into multiple possible emotions instead of only positive or negative sentiment.
