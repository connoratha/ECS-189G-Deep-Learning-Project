import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lowercase
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return tokens

def load_and_preprocess_data(directory):
    texts, labels = [], []
    for label_dir in ["pos", "neg"]:
        label_dir_path = os.path.join(directory, label_dir)
        for filename in os.listdir(label_dir_path):
            file_path = os.path.join(label_dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            texts.append(clean_text(text))
            labels.append(1 if label_dir == "pos" else 0)
    return texts, labels

# Example usage
train_texts, train_labels = load_and_preprocess_data('/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/stage_4_data/text_classification/train')
test_texts, test_labels = load_and_preprocess_data('/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/stage_4_data/text_classification/test')

def build_vocab(texts):
    # Flatten the list of token lists into a single list of tokens
    all_tokens = [token for text in texts for token in text]
    # Count tokens and sort by frequency
    vocab_counter = Counter(all_tokens)
    # Create vocab list (starting with special tokens)
    vocab = ['<pad>', '<unk>'] + [token for token, count in vocab_counter.items()]
    return {word: i for i, word in enumerate(vocab)}

vocab = build_vocab(train_texts)

def encode_texts(texts, vocab):
    encoded_texts = []
    for text in texts:
        encoded_text = [vocab.get(token, vocab['<unk>']) for token in text]
        encoded_texts.append(torch.tensor(encoded_text))
    return encoded_texts

encoded_train_texts = encode_texts(train_texts, vocab)
encoded_test_texts = encode_texts(test_texts, vocab)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    labels = torch.tensor(labels)
    return texts_padded, labels

train_dataset = TextDataset(encoded_train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

test_dataset = TextDataset(encoded_test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_output):
        # rnn_output: (batch_size, seq_length, hidden_dim)
        attention_weights = torch.softmax(self.attention(rnn_output).squeeze(2), dim=1)
        # (batch_size, seq_length)
        context_vector = torch.sum(attention_weights.unsqueeze(2) * rnn_output, dim=1)
        # (batch_size, hidden_dim)
        return context_vector

class TextClassRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(TextClassRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        rnn_output, _ = self.rnn(embedded)
        context_vector = self.attention(rnn_output)
        context_vector = self.dropout(context_vector)
        out = self.fc(context_vector)
        return out



def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels.float())
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Instantiate the model, criterion, and optimizer
model = TextClassRNN(len(vocab), embedding_dim=100, hidden_dim=256, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
epoch_losses = []

# Training loop with scheduler step and loss collection
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    scheduler.step()  # Update the learning rate
    epoch_losses.append(train_loss)  # Store the average loss for plotting
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')


def calculate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in loader:
            predictions = model(texts).squeeze(1)
            predicted_labels = torch.round(torch.sigmoid(predictions))
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    return correct / total

accuracy = calculate_accuracy(model, test_loader)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b', label='Training Loss')
plt.title('Training Convergence')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.show()