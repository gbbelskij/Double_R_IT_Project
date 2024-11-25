import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import CourseRecommendationModel
from model.train import train_model
from data.database_config import get_test_data
import torch

# TODO: осталось доделать функции получения различных выборок тестовых данных (а для этого надо их откуда-то взять)

def test_model(model):
    labels = get_courses()
    features = get_test_data()
    features = torch.tensor(features, dtype=torch.float32)
    labels = labels - 1

    model.eval()
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, axis=1).numpy()
    
    # Display results
    for i, prediction in enumerate(predictions):
        expected = int(labels[i] + 1)
        emoji = "✅" if (prediction + 1 == expected) else "⚠️"
        print(f"Predicted: {prediction + 1}; "
              f"Expected: {expected} {emoji}")

# TODO : сделать разделение на несколько выборок, найти где-то тестовые данные, без них никуда

model = CourseRecommendationModel()

train_model(model, epochs=100, learning_rate=0.01)

test_model(model)
