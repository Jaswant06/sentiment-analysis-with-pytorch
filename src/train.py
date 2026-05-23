from pathlib import Path
import re
import json
from collections import Counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    return text.split()


df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[["text", "sentiment"]].dropna()
df = df[df["sentiment"].isin(["positive", "negative"])]

df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

counter = Counter()

for text in train_texts:
    counter.update(tokenize(text))

vocab = {"<PAD>": 0, "<UNK>": 1}

for word, count in counter.items():
    if count >= 2:
        vocab[word] = len(vocab)


def encode(text, max_len=40):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        x = torch.tensor(encode(self.texts[index]), dtype=torch.long)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y


train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
)


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

print("Using device:", device)
print("Dataset size:", len(df))
print("Vocab size:", len(vocab))
print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))

model = SentimentLSTM(len(vocab)).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15
train_losses = []
val_accuracies = []
val_f1_scores = []

final_labels = []
final_preds = []

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        logits = model(texts)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in val_loader:
            texts = texts.to(device)

            logits = model(texts)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).int().cpu().numpy()

            all_preds.extend(predictions)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    val_accuracies.append(accuracy)
    val_f1_scores.append(f1)

    final_labels = all_labels
    final_preds = all_preds

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Loss: {average_loss:.4f} - "
        f"Val Accuracy: {accuracy:.4f} - "
        f"F1: {f1:.4f}"
    )

torch.save(model.state_dict(), MODELS_DIR / "sentiment_lstm.pth")

with open(MODELS_DIR / "vocab.json", "w") as f:
    json.dump(vocab, f)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_accuracies, marker="o")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "training_curves.png", bbox_inches="tight")
plt.close()

conf_matrix = confusion_matrix(final_labels, final_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["negative", "positive"],
    yticklabels=["negative", "positive"],
)

plt.title("Sentiment Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(PLOTS_DIR / "confusion_matrix.png", bbox_inches="tight")
plt.close()

print("Model saved to models/sentiment_lstm.pth")
print("Vocabulary saved to models/vocab.json")
print("Plots saved to plots/")