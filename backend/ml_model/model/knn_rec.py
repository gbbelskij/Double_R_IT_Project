from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNRecommender:
    def __init__(self, user_features, n_neighbors=5):
        self.user_features = user_features
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.knn.fit(user_features)
    
    def get_similar_users(self, user_id):
        user_vec = self.user_features[user_id].reshape(1, -1)
        _, indices = self.knn.kneighbors(user_vec)
        return indices[0]
    
    def get_candidate_courses(self, user_id, interactions):
        similar_users = self.get_similar_users(user_id)
        candidate_courses = interactions[interactions["user_id"].isin(similar_users)]["course_id"].unique()
        return candidate_courses