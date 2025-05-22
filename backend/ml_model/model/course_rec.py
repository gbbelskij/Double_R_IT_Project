from data_loading import load_data, load_train_data, prepare_features
from knn_rec import KNNRecommender
from two_tower_model import TwoTowerModel, train_model
from matrix_factorization import MatrixFactorization, train_matrix_factorization
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class RecommendationSystem:
    def __init__(self, n_recommendations=5, similarity_threshold=0.3, mf_weight=0.6):
        # Загружаем реальные данные для рекомендаций
        self.users, self.courses, self.interactions = load_data()
        self.user_features, self.course_features = prepare_features(self.users, self.courses)
        
        # Загружаем тренировочные данные для обучения модели
        self.train_users, self.train_courses, self.train_interactions = load_train_data()
        self.train_user_features, self.train_course_features = prepare_features(self.train_users, self.train_courses)
        
        # Подготовка маппинга id -> индекс для матричной факторизации
        self.user_id_to_idx = {id: idx for idx, id in enumerate(self.users['user_id'].unique())}
        self.course_id_to_idx = {id: idx for idx, id in enumerate(self.courses['course_id'].unique())}
        
        self.knn = KNNRecommender(self.user_features,
                                self.users,
                                similarity_threshold=similarity_threshold)
        
        # Initialize the Two Tower model
        self.model = TwoTowerModel(
            user_dim=self.train_user_features.shape[1],
            course_dim=self.train_course_features.shape[1],
            embedding_dim=64,
            tower_hidden_dims=[128, 64]
        )
        
        # Подготавливаем данные для обучения из тестовых данных
        self._prepare_training_data()
        self.model = train_model(
            self.model, 
            self.train_loader, 
            epochs=100,
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Матричная факторизация на реальных данных
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
        # Используем тестовые данные для обучения
        user_ids = self.train_interactions["user_id"].values
        course_ids = self.train_interactions["course_id"].values
        labels = self.train_interactions["liked"].values
        
        X_users = torch.tensor(self.train_user_features[user_ids], dtype=torch.float32)
        X_courses = torch.tensor(self.train_course_features[course_ids], dtype=torch.float32)
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
        
        # 3. Если нужно, дополняем рекомендациями от Two Tower модели
        if len(candidates) < self.n_recommendations * 2:
            count = self.n_recommendations * 2 - len(candidates)
            tt_courses = self._get_two_tower_candidates(
                user_id,
                count,
                exclude=np.concatenate([list(candidates), exclude])
            )
            candidates.update(tt_courses)
            print(50 * "=")
            print(f"Two Tower model recommendations:\n{tt_courses}")
            print(50 * "=")
        
        # Ранжируем кандидатов, комбинируя оценки от Two Tower и матричной факторизации
        ranked_candidates = self._rank_with_ensemble(user_id, list(candidates))

        print("=" * 50)
        print("Ранжирование кандидатов...")
        print("=" * 50)
        print("Топ-5 рекомендаций:")
        for c in ranked_candidates[:self.n_recommendations]:
            print(f"Курс: {c[0]}\tОбщий: {c[1]:.2f}\tMF: {c[2]:.2f}\tTT: {c[3]:.2f}")
        
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
    
    def _get_two_tower_candidates(self, user_id, count, exclude=[]):
        """Получить рекомендации от Two Tower модели"""
        # Адаптируем векторы пользователей к формату обученной модели
        user_vec = torch.tensor(self.user_features[user_id], 
                              dtype=torch.float32).unsqueeze(0)
        
        # Если размерности векторов не совпадают, нужно их адаптировать
        if user_vec.shape[1] != self.model.user_tower[0].in_features:
            print(f"Предупреждение: Размерности не совпадают. Используем заполнитель фичей.")
            # Создаем заполнитель нужной размерности
            user_vec = torch.zeros(1, self.model.user_tower[0].in_features, dtype=torch.float32)
            # Копируем доступные фичи
            min_dim = min(self.user_features[user_id].shape[0], self.model.user_tower[0].in_features)
            user_vec[0, :min_dim] = torch.tensor(self.user_features[user_id][:min_dim], dtype=torch.float32)
        
        self.model.eval()
        
        # Адаптируем размерности векторов курсов если нужно
        course_features = self.course_features
        if course_features.shape[1] != self.model.course_tower[0].in_features:
            print(f"Предупреждение: Размерности векторов курсов не совпадают. Адаптируем.")
            adapted_course_features = np.zeros((len(course_features), self.model.course_tower[0].in_features))
            min_dim = min(course_features.shape[1], self.model.course_tower[0].in_features)
            adapted_course_features[:, :min_dim] = course_features[:, :min_dim]
            course_features = adapted_course_features
        
        predictions = self.model.predict_all_courses(user_vec, course_features)
        
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
        """Ранжирует кандидатов, комбинируя оценки Two Tower модели и матричной факторизации"""
        user_vec = torch.tensor(self.user_features[user_id], 
                            dtype=torch.float32).unsqueeze(0)
        
        # Адаптируем размерность вектора пользователя если нужно
        if user_vec.shape[1] != self.model.user_tower[0].in_features:
            user_vec = torch.zeros(1, self.model.user_tower[0].in_features, dtype=torch.float32)
            min_dim = min(self.user_features[user_id].shape[0], self.model.user_tower[0].in_features)
            user_vec[0, :min_dim] = torch.tensor(self.user_features[user_id][:min_dim], dtype=torch.float32)
        
        self.model.eval()
        user_idx = self.user_id_to_idx[user_id]
        
        scores = []
        for course_id in candidate_courses:
            # Оценка от Two Tower модели
            course_idx = np.where(self.all_course_ids == course_id)[0][0]
            course_vec = torch.tensor(self.course_features[course_idx], 
                                    dtype=torch.float32).unsqueeze(0)
            
            # Адаптируем размерность вектора курса если нужно
            if course_vec.shape[1] != self.model.course_tower[0].in_features:
                adapted_course_vec = torch.zeros(1, self.model.course_tower[0].in_features, dtype=torch.float32)
                min_dim = min(course_vec.shape[1], self.model.course_tower[0].in_features)
                adapted_course_vec[0, :min_dim] = course_vec[0, :min_dim]
                course_vec = adapted_course_vec
            
            tt_score = self.model(user_vec, course_vec).item()
            
            # Оценка от матричной факторизации
            mf_course_idx = self.course_id_to_idx[course_id]
            mf_score = self.mf_model.predict_all_courses(user_idx, [mf_course_idx])
            mf_score = float(mf_score)
            
            # Взвешенная комбинация
            ensemble_score = (1 - self.mf_weight) * tt_score + self.mf_weight * mf_score
            
            scores.append((course_id, ensemble_score, mf_score, tt_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [[int(course_id), float(score), float(mf_score), float(tt_score)] for course_id, score, mf_score, tt_score in scores]

def main():
    # Новый параметр mf_weight
    system = RecommendationSystem(n_recommendations=5, similarity_threshold=0.4, mf_weight=0.3)
    user_id = 10

    system.recommend(user_id)

if __name__ == "__main__":
    main()