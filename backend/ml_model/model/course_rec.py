from data_loading import load_data, prepare_features
from knn_rec import KNNRecommender
from model import RecSysModel, train_model
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class RecommendationSystem:
    def __init__(self, n_recommendations=5, similarity_threshold=0.3):
        self.users, self.courses, self.interactions = load_data()
        self.user_features, self.course_features = prepare_features(self.users, self.courses)
        
        self.knn = KNNRecommender(self.user_features, 
                                similarity_threshold=similarity_threshold)
        
        self.model = RecSysModel(self.user_features.shape[1], 
                               self.course_features.shape[1])
        
        self._prepare_training_data()
        self.model = train_model(self.model, self.train_loader)
        
        self.n_recommendations = n_recommendations
        self.all_course_ids = self.courses['course_id'].unique()
    
    def _prepare_training_data(self):
        user_ids = self.interactions["user_id"].values
        course_ids = self.interactions["course_id"].values
        labels = self.interactions["liked"].values
        
        X_users = torch.tensor(self.user_features[user_ids], dtype=torch.float32)
        X_courses = torch.tensor(self.course_features[course_ids], dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_users, X_courses, y)
        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    def _get_liked_courses(self, similar_users):
        similar_user_ids = [user[0] for user in similar_users]
        liked = self.interactions[
            (self.interactions["user_id"].isin(similar_user_ids)) &
            (self.interactions["liked"] == 1)
        ]
        return liked["course_id"].unique()
    
    def recommend(self, user_id):
        # Получаем похожих пользователей
        similar_users = self.knn.get_similar_users(user_id)
        print(50 * "=")
        print(f"similar_users:\n{similar_users}")
        print(50 * "=")
        
        # Если нет похожих пользователей, используем только MLP
        if not similar_users:
            return self._mlp_based_recommendations(user_id)
        
        # Получаем курсы, которые понравились похожим пользователям
        candidate_courses = self._get_liked_courses(similar_users)
        print(50 * "=")
        print(f"courses liked by similar_users:\n{candidate_courses}")
        print(50 * "=")
        
        # Если кандидатов достаточно
        if len(candidate_courses) >= self.n_recommendations:
            return self._rank_and_select(user_id, candidate_courses)
            
        # Если кандидатов недостаточно, дополняем MLP-рекомендациями
        count = self.n_recommendations - len(candidate_courses)
        mlp_candidates = self._get_mlp_candidates(user_id, count, exclude=candidate_courses)
        combined = np.concatenate([candidate_courses, mlp_candidates])
        
        return self._rank_and_select(user_id, combined)
    
    def _mlp_based_recommendations(self, user_id):
        user_vec = torch.tensor(self.user_features[user_id], 
                              dtype=torch.float32).unsqueeze(0)
        predictions = self.model.predict_all_courses(user_vec, self.course_features)
        
        course_scores = list(zip(self.all_course_ids, predictions))
        course_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [int(course_id) for course_id, _ in course_scores[:self.n_recommendations]]
    
    def _get_mlp_candidates(self, user_id, count, exclude=[]):
        user_vec = torch.tensor(self.user_features[user_id], 
                              dtype=torch.float32).unsqueeze(0)
        predictions = self.model.predict_all_courses(user_vec, self.course_features)
        
        # Фильтруем исключенные курсы
        valid_indices = [i for i, cid in enumerate(self.all_course_ids) 
                       if (cid not in exclude)]
        valid_scores = predictions[valid_indices]
        valid_courses = self.all_course_ids[valid_indices]
        
        # Выбираем топ-N
        top_indices = np.argsort(valid_scores)[-count:][::-1]
        print(50 * "=")
        print(f"valid_courses[top_indices]:\n{valid_courses[top_indices]}")
        print(50 * "=")
        return valid_courses[top_indices]
    
    def _rank_and_select(self, user_id, candidate_courses):
        user_vec = torch.tensor(self.user_features[user_id], 
                              dtype=torch.float32).unsqueeze(0)
        
        scores = []
        for course_id in candidate_courses:
            course_idx = np.where(self.all_course_ids == course_id)[0][0]
            course_vec = torch.tensor(self.course_features[course_idx], 
                                    dtype=torch.float32).unsqueeze(0)
            score = self.model(user_vec, course_vec).item()
            scores.append((course_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [[int(course_id), float(score)] for course_id, score in scores[:self.n_recommendations]]

def main():
    system = RecommendationSystem(n_recommendations=5, similarity_threshold=0.4)
    user_id = 2
    
    # Пример использования
    print("=" * 50)
    print(f"Рекомендации для пользователя {user_id}:")
    print("=" * 50)
    for c in system.recommend(user_id):
        print(f"corse: {c[0]}, rating: {c[1]}")

if __name__ == "__main__":
    main()