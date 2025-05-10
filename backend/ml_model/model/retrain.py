# retrain.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import CourseRecommender
from save_load import load_model, save_model

# Загрузка модели
model = CourseRecommender(num_tags=10)  # Замените на реальное количество тегов
model = load_model(model, "course_recommender.pth")

# Новые данные для дообучения (пример)
new_course_tensor = torch.tensor([[0, 1, 2], [3, 4, 5]])  # Замените на реальные данные
new_user_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])    # Замените на реальные данные
new_y_true = torch.FloatTensor([[1.0, 0.0], [0.0, 1.0]])  # Замените на реальные данные

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Дообучение
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(new_course_tensor, new_user_tensor)
    loss = criterion(outputs, new_y_true)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Сохранение обновленной модели
save_model(model, "course_recommender_retrained.pth")