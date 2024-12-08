import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import CourseRecommendationModel
from data.database_config import get_test_data
from save_load import save_model, load_model
from train import train_model

model_save_path = "backend/ml_model/data/model/model.pt"

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


if __name__ == '__main__':
    model = CourseRecommendationModel()
    train_model(model, 1000, 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    load_model(model, optimizer, model_save_path)
    model.train()
    train_model(model, 10000, 0.001)
    model.eval()
    test_model(model)