import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re


# Read and preprocess text data
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        jokes = [line.split(',', 1)[1].strip('"') for line in lines]  # Adjust based on your dataset format
        text = " ".join(jokes).lower()  # Join all jokes into a single text string
        return text

text = read_data('/Users/devsingh/Desktop/Winter_2024/ECS189G/Deep_Learning/stage_4_data/text_generation/data')

# Tokenization and sequence creation
tokens = re.findall(r'\b\w+\b', text)
vocab = set(tokens)
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
sequences = [word_to_idx[word] for word in tokens]

# Create dataset
seq_length = 30
X, y = [], []
for i in range(len(sequences) - seq_length):
    X.append(sequences[i:i + seq_length])
    y.append(sequences[i + seq_length])
X = np.array(X)
y = np.array(y)


# Define model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        pred = self.fc(output[:, -1, :])
        return pred


# Hyperparameters
hidden_dim = 128
output_size = vocab_size
batch_size = 128
epochs = 20
lr = 0.001


# DataLoader
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def accuracy(predictions, targets):
    _, predicted_indices = predictions.max(1)  # Get the index of the max log-probability
    correct = (predicted_indices == targets).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc

# Initialize model, loss, and optimizer
model = RNNModel(vocab_size, hidden_dim, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

total_loss = 0
total_acc = 0
num_batches = 0


# Training loop
model.train()
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)  # Assume accuracy function is defined as before
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    # Optionally, print loss and accuracy for each epoch
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches}')

# Compute total accuracy after all epochs
total_accuracy = total_acc / num_batches * 100
print(f'Total Accuracy after {epochs} epochs: {total_accuracy}%')

# Simplified text generation function
def generate_text(start_seq, length=50, idx_to_word=idx_to_word):
    model.eval()
    generated_text = start_seq
    for _ in range(length):
        input_seq = [word_to_idx.get(word, 0) for word in start_seq.split()]
        input_tensor = torch.LongTensor(input_seq).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        predicted_index = output.argmax(1).item()
        predicted_word = idx_to_word[predicted_index]
        generated_text += ' ' + predicted_word
        start_seq = ' '.join(start_seq.split()[1:] + [predicted_word])
    return generated_text


# Example usage
print(generate_text("what did the", 10))
