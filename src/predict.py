from pathlib import Path
import re
import json

import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "sentiment_lstm.pth"
VOCAB_PATH = MODELS_DIR / "vocab.json"


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    return text.split()


def encode(text, vocab, max_len=40):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=64,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden_combined = torch.cat(
            (hidden_forward, hidden_backward),
            dim=1,
        )

        hidden_combined = self.dropout(hidden_combined)
        logits = self.fc(hidden_combined)

        return logits.squeeze(1)


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

model = SentimentLSTM(len(vocab)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Sentiment Analysis Predictor")
print("Type a sentence. Type 'quit' to stop.\n")

while True:
    text = input("Enter text: ")

    if text.lower() == "quit":
        break

    encoded = encode(text, vocab)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    if probability >= 0.5:
        label = "positive"
        confidence = probability
    else:
        label = "negative"
        confidence = 1 - probability

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%\n")