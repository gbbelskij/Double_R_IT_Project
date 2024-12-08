from model import CourseRecommendationModel
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.database_config import get_test_data

# TODO: осталось доделать функции получения различных выборок тестовых данных (а для этого надо их откуда-то взять)

model_save_path = "backend/ml_model/data/model/model.pt"

def train_model(model, epochs, learning_rate):
    features, labels = get_test_data()
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels - 1, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


if __name__ == '__main__':
    model = CourseRecommendationModel()
    train_model(model, epochs=10000, learning_rate=0.01)
    model.eval()
    torch.save(model.state_dict(), model_save_path)
    
