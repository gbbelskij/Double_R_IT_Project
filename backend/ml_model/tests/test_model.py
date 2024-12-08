import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import CourseRecommendationModel
from model.train import train_model
from data.database_config import get_test_data

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
    train_model(model, epochs=10000, learning_rate=0.01)
    test_model(model)

