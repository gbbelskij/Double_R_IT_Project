import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class KNNRecommender:
    def __init__(self, user_features, users_df, n_neighbors=5, similarity_threshold=1.0):
        """
        user_features - исходные one-hot encoded фичи (для совместимости)
        users_df - DataFrame с исходными признаками (user_id, Lang, Department, Add tags, Lvl)
        """
        self.users_df = users_df
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.weights = {'Lang': 1.0, 'Department': 2.0, 'Add tags': 0.5, 'Lvl': 0.2}
        
        # Закодируем категориальные признаки
        self.encoders = {}
        self.encoded_data = {}
        
        for col in ['Lang', 'Department', 'Add tags', 'Lvl']:
            le = LabelEncoder()
            self.encoded_data[col] = le.fit_transform(users_df[col])
            self.encoders[col] = le

        print("=" * 50)
        print("Инициализация KNNRecommender...")
        print(f"- Веса признаков: {self.weights}")
        print(f"- Порог схожести: {similarity_threshold}")
        print("=" * 50)
    
    def _calculate_similarity(self, user1_idx, user2_idx):
        """Вычисляет взвешенную схожесть между двумя пользователями"""
        total_similarity = 0.0
        total_weight = 0.0
        
        for col in ['Lang', 'Department', 'Add tags', 'Lvl']:
            weight = self.weights[col]
            if self.encoded_data[col][user1_idx] == self.encoded_data[col][user2_idx]:
                total_similarity += weight
            total_weight += weight
        
        return total_similarity / total_weight if total_weight > 0 else 0
    
    def get_similar_users(self, user_id):
        """Возвращает список кортежей (индекс пользователя, схожесть)"""
        # Находим индексы всех пользователей, кроме текущего
        all_indices = [i for i in range(len(self.users_df)) if i != user_id]
        
        # Вычисляем схожесть с каждым пользователем
        similarities = []
        for idx in all_indices:
            sim = self._calculate_similarity(user_id, idx)
            similarities.append((idx, sim))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Фильтруем по порогу схожести и берем топ-N
        filtered = [(idx, sim) for idx, sim in similarities 
                  if sim >= self.similarity_threshold][:self.n_neighbors]
        
        print("="*50)
        print(f"Similar users for user {user_id}:")
        for idx, sim in filtered:
            print(f"User {idx}: {sim:.2f} similarity")
        print("="*50)
        
        return filtered