import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_courses, n_factors=20):
        super(MatrixFactorization, self).__init__()
        # Инициализируем матрицы латентных факторов
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.course_factors = nn.Embedding(n_courses, n_factors)
        
        # Инициализируем смещения для пользователей и курсов
        self.user_biases = nn.Embedding(n_users, 1)
        self.course_biases = nn.Embedding(n_courses, 1)
        
        # Общее смещение (глобальное среднее)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Инициализация весов
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.course_factors.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.course_biases.weight)
    
    def forward(self, user_indices, course_indices):
        # Получаем факторы пользователей и курсов
        user_f = self.user_factors(user_indices)
        course_f = self.course_factors(course_indices)
        
        # Вычисляем смещения
        user_b = self.user_biases(user_indices)
        course_b = self.course_biases(course_indices)
        
        # Вычисляем скалярное произведение латентных факторов
        dot_product = torch.sum(user_f * course_f, dim=1, keepdim=True)
        
        # Полное предсказание = глобальное смещение + смещение пользователя + 
        # смещение курса + скалярное произведение факторов
        prediction = self.global_bias + user_b + course_b + dot_product
        
        return torch.sigmoid(prediction)
    
    def predict_all_courses(self, user_id, course_ids):
        """Предсказать рейтинги для конкретного пользователя и всех курсов"""
        user_tensor = torch.tensor([user_id] * len(course_ids), dtype=torch.long)
        course_tensor = torch.tensor(course_ids, dtype=torch.long)
        
        with torch.no_grad():
            predictions = self(user_tensor, course_tensor)
        
        return predictions.squeeze().numpy()


def train_matrix_factorization(interactions, n_users, n_courses, epochs=100, lr=0.01, 
                               n_factors=20, reg_param=0.01):
    """Обучение модели матричной факторизации"""
    print("=" * 50)
    print("Обучение MatrixFactorization...")
    # Создаем модель
    model = MatrixFactorization(n_users, n_courses, n_factors)
    
    # Готовим данные
    user_ids = torch.tensor(interactions["user_id"].values, dtype=torch.long)
    course_ids = torch.tensor(interactions["course_id"].values, dtype=torch.long)
    ratings = torch.tensor(interactions["liked"].values, dtype=torch.float32).view(-1, 1)
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_param)
    criterion = nn.BCELoss()
    
    # Обучаем модель
    for epoch in range(epochs):
        # Прямой проход
        optimizer.zero_grad()
        predictions = model(user_ids, course_ids)
        
        # Вычисляем потери
        loss = criterion(predictions, ratings)
        
        # Обратное распространение
        loss.backward()
        optimizer.step()
        
        # Логируем прогресс
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    print("Обучение завершено")
    print("=" * 50)
    return model