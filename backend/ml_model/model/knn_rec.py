from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNRecommender:
    def __init__(self, user_features, n_neighbors=2, similarity_threshold=0.5):
        self.user_features = user_features
        self.similarity_threshold = similarity_threshold
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.knn.fit(user_features)
    
    def get_similar_users(self, user_id):
        user_vec = self.user_features[user_id].reshape(1, -1)
        distances, indices = self.knn.kneighbors(user_vec)
        
        # Конвертируем расстояния в схожесть (1 - distance)
        similarities = 1 - distances[0]
        # Фильтруем по порогу схожести
        filtered = [(idx, sim) for idx, sim in zip(indices[0], similarities) 
                  if sim > self.similarity_threshold and idx != user_id]
        print(50* "=")
        print(f"filtered:\n{filtered}")
        print(50* "=")
        
        return filtered  # Возвращаем кортежи (индекс, схожесть)