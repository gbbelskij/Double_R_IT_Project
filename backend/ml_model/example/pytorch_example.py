import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Course classes


class Course:
    FRONT = 1
    BACK = 2
    ML = 3
    SQL = 4


# Dictionary from number to course name
courses = {
    1: 'Frontend',
    2: 'Backend',
    3: 'ML',
    4: 'SQL'
}

# Neural Network


class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super(ImprovedNeuralNetwork, self).__init__()
        # First layer: 4 inputs -> 8 hidden neurons
        self.fc1 = nn.Linear(4, 8)      # Linear layer
        self.relu = nn.ReLU()           # Activation function
        self.dropout = nn.Dropout(0.2)  # Regularization

        # Second layer: 8 hidden -> 4 outputs
        self.fc2 = nn.Linear(8, 4)          # Linear layer
        self.softmax = nn.Softmax(dim=1)    # For probabilistic output

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        output = self.softmax(output)
        return output

# Function to normalize data


def normalize_data(data):
    data = np.array(data)
    features = data[:, :4]  # Get first 4 columns
    labels = data[:, 4]     # Get last column (expected output)

    # Normalization of numeric data
    # Normalize numeric data by default formula
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    return features, labels

# Training the model


def train_model(model, data, epochs, learning_rate):
    features, labels = normalize_data(data)                 # Normalize data
    # Make float32 tensor
    features = torch.tensor(features, dtype=torch.float32)
    # Targets start from 0
    labels = torch.tensor(labels - 1, dtype=torch.long)

    # Adam optimizer with learning rate (just needed)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()   # Loss function (to count error)

    for epoch in range(epochs):
        model.train()           # Set model to training mode
        optimizer.zero_grad()   # Clear gradients

        # Forward pass
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()         # Backpropagation (алгоритм обратного распр. ошибки)

        # Optimization
        optimizer.step()        # Update weights

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Testing the model


def test_model(model, data):
    features, labels = normalize_data(data)
    features = torch.tensor(features, dtype=torch.float32)
    labels = labels - 1  # Targets start from 0

    model.eval()    # Set model to evaluation mode (without changing weights and counting gradients)
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, axis=1).numpy()

    # Display results
    for i, prediction in enumerate(predictions):
        expected = int(labels[i] + 1)
        emoji = "✅" if (prediction + 1 == expected) else "⚠️"
        print(f"Predicted: {courses[prediction + 1].ljust(10)}; "
              f"Expected: {courses[expected].ljust(10)} {emoji}")


# Training data
epoch = [
    (1, 0, 1, 0, Course.FRONT),
    (1, 0, 1, 1, Course.SQL),
    (1, 0, 1, 0, Course.FRONT),
    (0, 1, 0, 1, Course.BACK),
    (0, 1, 0, 1, Course.BACK),
    (0, 1, 0, 1, Course.SQL),
    (0, 1, 0, 0, Course.ML),
    (0, 1, 1, 0, Course.ML),
    (0, 0, 0, 1, Course.SQL),
    (1, 0, 0, 1, Course.SQL),
    (1, 1, 1, 1, Course.SQL),
    (1, 1, 1, 0, Course.FRONT),
]

# Creating and training the model
model = ImprovedNeuralNetwork()
train_model(model, epoch, epochs=100, learning_rate=0.01)

# Testing the model
test_model(model, epoch)
