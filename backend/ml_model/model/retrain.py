import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import CourseRecommendationModel
from data.database_config import get_test_data
from save_load import save_model, load_model
from train import train_model
from scripts.data_preprocessing import dp_from_json

MODEL_SAVE_PATH = "backend/ml_model/data/model/model.pt"

def test_model(model: CourseRecommendationModel) -> None:
    features, labels = get_test_data()
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels - 1, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, axis=1)

    for i, prediction in enumerate(predictions):
        expected = labels[i]
        emoji = "✅" if prediction == expected else "⚠️"
        print(f"Predicted: {prediction + 1}; Expected: {expected + 1} {emoji}")

def fl_from_json(user_data: str):
    user_data = user_data.replace("'", '"')

    preprocessed_data = json.loads(dp_from_json(user_data))
    arr = [i['0'] for i in preprocessed_data.values()]
    
    feature = arr[:-1]
    label = arr[-1]
    return feature, label

def retrain_model(model, data: str, epochs = 1000, learning_rate = 0.001):
    feature, label = fl_from_json(data)
    model.train()
    
    feature = torch.tensor([feature], dtype=torch.float32)
    label = torch.tensor([label - 1], dtype=torch.long)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(feature)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            
        model.eval()
        save_model(model, optimizer, MODEL_SAVE_PATH)

if __name__ == '__main__':
    model = CourseRecommendationModel()
    train_model(model, 10000, 0.001)
    
    # Create new model and laod weights into it
    model = CourseRecommendationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    load_model(model, optimizer, MODEL_SAVE_PATH)
    
    # Create new data
    user_data = {
        "Age": [24],
        "Experience": [5],
        "Role": ["QA Engineer"],
        "Department": ["Sales"],
        "Answers": ["(1, 0, 0, 0, 0, 0, 0, 0, 1, 0)"],
        "Course": [24]
    }
    
    # Aftertraining model on new data
    model.train()
    retrain_model(model, str(user_data))
    model.eval()
    
    # Test retrained model
    test_model(model)
