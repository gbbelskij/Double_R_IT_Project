from data_loading import load_data, prepare_features
from knn_rec import KNNRecommender
from model import RecSysModel, train_model
from matrix_factorization import MatrixFactorization, train_matrix_factorization  # Новая строка
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class RecommendationSystem:
    def __init__(self, n_recommendations=5, similarity_threshold=0.3, mf_weight=0.3):  # Добавлен параметр mf_weight
        self.users, self.courses, self.interactions = load_data()
        self.user_features, self.course_features = prepare_features(self.users, self.courses)
        
        # Подготовка маппинга id -> индекс для матричной факторизации
        self.user_id_to_idx = {id: idx for idx, id in enumerate(self.users['user_id'].unique())}
        self.course_id_to_idx = {id: idx for idx, id in enumerate(self.courses['course_id'].unique())}
        
        self.knn = KNNRecommender(self.user_features,
                                self.users,
                                similarity_threshold=similarity_threshold)
        
        self.model = RecSysModel(self.user_features.shape[1], 
                               self.course_features.shape[1])
        
        self._prepare_training_data()
        self.model = train_model(self.model, self.train_loader)
        
        # Матричная факторизация - НОВЫЙ КОД
        n_users = len(self.users['user_id'].unique())
        n_courses = len(self.courses['course_id'].unique())
        self.mf_model = train_matrix_factorization(
            self.interactions, n_users, n_courses, n_factors=15, epochs=50
        )
        
        # Вес матричной факторизации при ансамблировании
        self.mf_weight = mf_weight
        
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
    
    def _get_user_liked_courses(self, user_id):
        """Получить курсы, которые уже понравились пользователю"""
        liked = self.interactions[
            (self.interactions["user_id"] == user_id) & 
            (self.interactions["liked"] == 1)
        ]
        return liked["course_id"].unique()
    
    def _get_disliked_courses(self, user_id):
        disliked = self.interactions[
            (self.interactions["user_id"] == user_id) &
            (self.interactions["liked"] == 0)
        ]
        return disliked["course_id"].unique()
    
    def recommend(self, user_id):
        """Рекомендует курсы, комбинируя все подходы"""
        # Получаем понравившиеся курсы для исключения
        liked_courses = self._get_user_liked_courses(user_id)
        disliked_courses = self._get_disliked_courses(user_id)
        exclude = np.concatenate([liked_courses, disliked_courses])
        
        # Получаем похожих пользователей через KNN
        similar_users = self.knn.get_similar_users(user_id)
        print(50 * "=")
        print(f"similar_users:\n{similar_users}")
        print(50 * "=")
        
        # Формируем базовый набор кандидатов
        candidates = set()
        
        # 1. Добавляем рекомендации от KNN если есть похожие пользователи
        if similar_users:
            knn_courses = self._get_liked_courses(similar_users)
            knn_courses = [c for c in knn_courses if c not in exclude]
            candidates.update(knn_courses)
            print(50 * "=")
            print(f"courses liked by similar_users:\n{knn_courses}")
            print(50 * "=")
        
        # 2. Добавляем рекомендации от матричной факторизации
        mf_count = max(5, self.n_recommendations)  # Получаем больше кандидатов для ранжирования
        mf_courses = self._get_mf_recommendations(
            user_id, 
            count=mf_count,
            exclude=exclude
        )
        candidates.update(mf_courses)
        print(50 * "=")
        print(f"Matrix factorization recommendations:\n{mf_courses}")
        print(50 * "=")
        
        # 3. Если нужно, дополняем рекомендациями от MLP
        if len(candidates) < self.n_recommendations * 2:
            count = self.n_recommendations * 2 - len(candidates)
            mlp_courses = self._get_mlp_candidates(
                user_id,
                count,
                exclude=np.concatenate([list(candidates), exclude])
            )
            candidates.update(mlp_courses)
        
        # Ранжируем кандидатов, комбинируя оценки от MLP и матричной факторизации
        ranked_candidates = self._rank_with_ensemble(user_id, list(candidates))

        print("=" * 50)
        print("Ранжирование кандидатов...")
        print("=" * 50)
        print("Топ-5 рекомендаций:")
        for c in ranked_candidates[:self.n_recommendations]:
            print(f"Курс: {c[0]}\tОбщий: {c[1]:.2f}\tMF: {c[2]:.2f}\tMLP: {c[3]:.2f}")
        
        return ranked_candidates[:self.n_recommendations]
    
    def _get_mf_recommendations(self, user_id, count=10, exclude=[]):
        """Получить рекомендации на основе матричной факторизации"""
        # Переводим ID пользователя в индекс
        user_idx = self.user_id_to_idx[user_id]
        
        # Получаем все индексы курсов кроме исключенных
        course_indices = [self.course_id_to_idx[cid] for cid in self.all_course_ids 
                         if cid not in exclude]
        course_ids = [cid for cid in self.all_course_ids if cid not in exclude]
        
        # Предсказываем рейтинги
        predictions = self.mf_model.predict_all_courses(user_idx, course_indices)
        
        # Сортируем и возвращаем топ-N
        course_scores = list(zip(course_ids, predictions))
        course_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [cid for cid, _ in course_scores[:count]]
    
    def _mlp_based_recommendations(self, user_id, exclude=None):
        if exclude is None:
            exclude = []
        user_vec = torch.tensor(self.user_features[user_id], dtype=torch.float32).unsqueeze(0)
        predictions = self.model.predict_all_courses(user_vec, self.course_features)
        
        course_scores = list(zip(self.all_course_ids, predictions))
        course_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Фильтруем исключенные курсы
        filtered = [course for course in course_scores if course[0] not in exclude]
        return [int(course_id) for course_id, _ in filtered[:self.n_recommendations]]
    
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
    
    def _rank_with_ensemble(self, user_id, candidate_courses):
        """Ранжирует кандидатов, комбинируя оценки MLP и матричной факторизации"""
        user_vec = torch.tensor(self.user_features[user_id], 
                            dtype=torch.float32).unsqueeze(0)
        user_idx = self.user_id_to_idx[user_id]
        
        scores = []
        for course_id in candidate_courses:
            # Оценка от MLP
            course_idx = np.where(self.all_course_ids == course_id)[0][0]
            course_vec = torch.tensor(self.course_features[course_idx], 
                                    dtype=torch.float32).unsqueeze(0)
            mlp_score = self.model(user_vec, course_vec).item()
            
            # Оценка от матричной факторизации
            mf_course_idx = self.course_id_to_idx[course_id]
            mf_score = self.mf_model.predict_all_courses(user_idx, [mf_course_idx])
            mf_score = float(mf_score)
            
            # Взвешенная комбинация
            ensemble_score = (1 - self.mf_weight) * mlp_score + self.mf_weight * mf_score
            
            scores.append((course_id, ensemble_score, mf_score, mlp_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [[int(course_id), float(score), float(mf_score), float(mlp_score)] for course_id, score, mf_score, mlp_score in scores]
    
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
    # Новый параметр mf_weight
    system = RecommendationSystem(n_recommendations=5, similarity_threshold=0.4, mf_weight=0.3)
    user_id = 0

    system.recommend(user_id)

if __name__ == "__main__":
    main()