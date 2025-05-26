import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class KNNRecommender:
    def __init__(self, users_df, n_neighbors=5, similarity_threshold=1.0):
        """
        users_df - DataFrame с исходными признаками (user_id, preferred_tags)
        """
        users_df = users_df.copy()
        users_df['user_id'] = users_df['user_id'].astype(str)
        # Преобразуем списки тегов в строки для кодирования
        if 'preferred_tags' in users_df.columns:
            users_df['preferred_tags_str'] = users_df['preferred_tags'].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))
        else:
            users_df['preferred_tags_str'] = ''
        self.user_id_to_idx = {str(user_id): idx for idx, user_id in enumerate(users_df['user_id'])}
        self.users_df = users_df.reset_index(drop=True)
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.weights = {'preferred_tags_str': 1.0}
        self.encoders = {}
        self.encoded_data = {}

        for col in ['preferred_tags_str']:
            le = LabelEncoder()
            self.encoded_data[col] = le.fit_transform(users_df[col])
            self.encoders[col] = le

        print("=" * 50)
        print("Инициализация KNNRecommender...")
        print(f"- Веса признаков: {self.weights}")
        print(f"- Порог схожести: {similarity_threshold}")
        print("=" * 50)

    def _calculate_similarity(self, user1_id, user2_id):
        user1_idx = self.user_id_to_idx[user1_id]
        user2_idx = self.user_id_to_idx[user2_id]
        total_similarity = 0.0
        total_weight = 0.0
        for col in ['preferred_tags_str']:
            weight = self.weights[col]
            if self.users_df[col].iloc[user1_idx] == self.users_df[col].iloc[user2_idx]:
                total_similarity += weight
            total_weight += weight
        return total_similarity / total_weight if total_weight > 0 else 0

    def get_similar_users(self, user_id):
        user_id = str(user_id)
        if user_id not in self.user_id_to_idx:
            print(f"User {user_id} not found!")
            return []
        all_user_ids = [uid for uid in self.users_df['user_id'] if uid != user_id]
        similarities = []
        for other_id in all_user_ids:
            sim = self._calculate_similarity(user_id, other_id)
            similarities.append((other_id, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        filtered = [(uid, sim) for uid, sim in similarities if sim >= self.similarity_threshold][:self.n_neighbors]
        return filtered
