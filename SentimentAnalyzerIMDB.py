from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from utils.utils import Utils

# Initialize the Utils class
utils = Utils()

# Set the device for GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


NUM_WORDS = 10000  # Limit for vocabulary
# Prompt user for model parameters Ex: 200, 600, 400, 324, 14
seq_len = int(input("Define the sequence length for training: \n"))
batch_size = int(input("Define the batch size for training: \n"))
embed_size = int(input("Define the embed size for training: \n"))
lstm_size = int(input("Define the LSTM size for training: \n"))
epochs = int(input("Define the epochs number for training: \n"))

# Load and prepare the dataset
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=NUM_WORDS)
train_data = pad_sequences(train_data, maxlen=seq_len, padding='post')
test_data = pad_sequences(test_data, maxlen=seq_len, padding='post')

# Split the training data into training and validation sets
SPLIT_FRAC = 0.8
train_x, val_x, train_y, val_y = train_test_split(
    train_data, train_labels, train_size=SPLIT_FRAC)

# Show information about the dataset
print("[Model informations]")
print(f"Training set: {len(train_x)}")
print(f"Evaluation set: {len(val_x)}")
print(f"Sequence length: {seq_len}")
print(f"Batch Size: {batch_size}")
print(f"Embed size: {embed_size}")
print(f"LSTM size: {lstm_size}")

# Convert data to PyTorch tensors and move them to GPU
train_x_tensor = torch.tensor(train_x, dtype=torch.long).to(device)
train_y_tensor = torch.tensor(
    train_y, dtype=torch.float32).unsqueeze(1).to(device)
val_x_tensor = torch.tensor(val_x, dtype=torch.long).to(device)
val_y_tensor = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1).to(device)
test_x_tensor = torch.tensor(test_data, dtype=torch.long).to(device)
test_y_tensor = torch.tensor(
    test_labels, dtype=torch.float32).unsqueeze(1).to(device)

# Create DataLoader for batching the data
train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the network


class CNN_LSTM_Model(nn.Module):
    def __init__(self, NUM_WORDS, embed_size, lstm_size):
        super(CNN_LSTM_Model, self).__init__()
        self.embedding = nn.Embedding(NUM_WORDS, embed_size)

        # Define convolutional layers
        self.conv1d_1 = nn.Conv1d(embed_size, 256, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv1d_2 = nn.Conv1d(256, 128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            128, lstm_size, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        # Dropout and activation function
        self.dropout = nn.Dropout(0.7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)

        # Permute for convolutional layer input
        x = x.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)

        # Convolutional and pooling layers
        x = self.conv1d_1(x)
        x = self.pool1(x)
        x = self.conv1d_2(x)
        x = self.pool2(x)

        # Permute for LSTM input
        x = x.permute(0, 2, 1)  # (batch_size, conv_output_len, 128)

        # LSTM layer
        x, _ = self.lstm(x)

        # Get the last output of the sequence
        x = x[:, -1, :]

        # Apply dropout
        x = self.dropout(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = self.fc5(x)

        # Final activation
        x = self.sigmoid(x)
        return x


# Initialize lists to store predictions and labels
all_preds = []
all_labels = []

# Initialize model and move it to GPU
model = CNN_LSTM_Model(NUM_WORDS, embed_size, lstm_size).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

# Lists to track loss and accuracy
train_loss = []
val_loss = []
train_acc = []
val_acc = []

# Loop to train the model
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    epoch_train_loss = 0
    correct_train = 0
    total_train = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(
            device)  # Move batches to GPU
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(x_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Calculate accuracy on training set
        preds = (outputs > 0.5).float()
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
        epoch_train_loss += loss.item()

    train_loss.append(epoch_train_loss / len(train_loader))
    train_acc.append(correct_train / total_train)

    # Validation
    model.eval()
    epoch_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # Disable gradient tracking
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(
                device)  # Move batches to GPU
            outputs = model(x_val)
            loss = criterion(outputs, y_val)  # Compute validation loss
            epoch_val_loss += loss.item()

            # Calculate accuracy on validation set
            preds = (outputs > 0.5).float()
            correct_val += (preds == y_val).sum().item()
            total_val += y_val.size(0)

    val_loss.append(epoch_val_loss / len(val_loader))
    val_acc.append(correct_val / total_val)

    # Print training and validation stats
    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {
              train_loss[-1]:.4f}, Train Acc: {train_acc[-1] * 100:.4f}%, "
          f"Validation Loss: {val_loss[-1]:.4f}, Validation Acc: {val_acc[-1] * 100:.4f}%")

# Evaluate model on the test data
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(
            device)  # Move batches to GPU
        outputs = model(x_test)
        test_loss += criterion(outputs, y_test).item()  # Compute test loss
        preds = (outputs > 0.5).float()
        correct += (preds == y_test).sum().item()

        # Append predictions and true labels for later analysis
        all_preds.append(preds.cpu())
        all_labels.append(y_test.cpu())

# Concatenate predictions and labels
preds_tensor = torch.cat(all_preds)
labels_tensor = torch.cat(all_labels)

# Calculate test accuracy
test_acc = correct / len(test_dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# Plot training and validation loss
utils.plot_loss(train_loss, val_loss)
# Plot training and validation accuracy
utils.plot_accuracy(train_acc, val_acc)
# Plot the class distribution of the training labels
utils.plot_class_distribution(train_labels)
# Plot error histogram and confusion matrix for model evaluation
utils.plot_error_histogram(preds_tensor, labels_tensor)
utils.plot_confusion_matrix(preds_tensor, labels_tensor)
