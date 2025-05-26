from backend.ml_model.model.data_loading import load_data, load_train_data, prepare_features
from backend.ml_model.model.knn_rec import KNNRecommender
from backend.ml_model.model.two_tower_model import TwoTowerModel, train_model
# from matrix_factorization import MatrixFactorization, train_matrix_factorization
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from backend.database.User import Course


class RecommendationSystem:
    def __init__(self, app, n_recommendations=5, similarity_threshold=0.3, mf_weight=0.6):
        # Загружаем реальные данные для рекомендаций
        self.users_df, self.courses_df, self.interactions = load_data(app)
        self.user_features, self.course_features = prepare_features(self.users_df, self.courses_df)
        self.knn = KNNRecommender(self.users_df, n_neighbors=5, similarity_threshold=similarity_threshold)
        self.all_course_ids = set(self.courses_df['course_id'])

        # Загружаем тренировочные данные
        # (self.train_users_raw, self.train_courses_raw,
        #  self.train_users, self.train_courses, self.train_interactions) = load_train_data()

        # Признаки для нейросетей/MLB
        # self.user_features, self.course_features = prepare_features(self.users, self.courses)
        # self.train_user_features, self.train_course_features = prepare_features(self.train_users, self.train_courses)

        # KNN работает с сырыми данными (train_users_raw)
        
        # Подготовка маппинга id -> индекс для матричной факторизации
        self.user_id_to_idx = {id: idx for idx, id in enumerate(self.users_df['user_id'].unique())}
        self.course_id_to_idx = {id: idx for idx, id in enumerate(self.courses_df['course_id'].unique())}
        # self.knn = KNNRecommender(self.train_user_features, self.train_users_raw, similarity_threshold=similarity_threshold)

        
        # Initialize the Two Tower model
        self.model = TwoTowerModel(
            user_dim=self.user_features.shape[1],
            course_dim=self.course_features.shape[1],
            embedding_dim=64,
            tower_hidden_dims=[128, 64]
        )
                
        # Не вызываем функции, связанные с interactions:
        # self._prepare_training_data()
        # self.model = train_model(...)
        # self.mf_model = train_matrix_factorization(...)
        
        self.mf_weight = mf_weight
        self.n_recommendations = n_recommendations
        with app.app_context():
            self.all_course_ids = np.array([str(c.course_id) for c in Course.query.all()])
    
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
        """Рекомендует курсы без использования interactions"""
        # НЕ вызываем функции, связанные с interactions!
        # liked_courses = self._get_user_liked_courses(user_id)
        # disliked_courses = self._get_disliked_courses(user_id)
        # exclude = np.concatenate([liked_courses, disliked_courses])
        
        similar_users = self.knn.get_similar_users(user_id)
        print(50 * "=")
        print(f"similar_users:\n{similar_users}")
        print(50 * "=")
        
        # Формируем базовый набор кандидатов — все курсы
        candidates = set(self.all_course_ids)
        
        # НЕ используем knn_courses, mf_courses, tt_courses, связанные с interactions
        
        # Ранжируем кандидатов (можно оставить только Two Tower или KNN)
        valid_candidates = [c for c in candidates if c in self.all_course_ids]
        ranked_candidates = self._rank_with_ensemble(user_id, valid_candidates)
        print("=" * 50)
        print("Ранжирование кандидатов...")
        print("=" * 50)
        print("Топ-5 рекомендаций:")
        
        for c in ranked_candidates[:self.n_recommendations]:
            print(f"Курс: {c[0]}\tОбщий: {c[1]:.2f}\tMF: {c[2]:.2f}\tTT: {c[3]:.2f}")
        
        return ranked_candidates[:self.n_recommendations]
    
    def _get_mf_recommendations(self, user_id, count=10, exclude=[]):
        user_idx = self.user_id_to_idx[user_id]
        course_indices = [self.course_id_to_idx[cid] for cid in self.all_course_ids 
                         if cid not in exclude]
        course_ids = [cid for cid in self.all_course_ids if cid not in exclude]
        predictions = np.zeros(len(course_indices))  # Заглушка
        course_scores = list(zip(course_ids, predictions))
        course_scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in course_scores[:count]]
    
    def _get_two_tower_candidates(self, user_id, count, exclude=[]):
        user_vec = torch.tensor(self.user_features[user_id], 
                              dtype=torch.float32).unsqueeze(0)
        if user_vec.shape[1] != self.model.user_tower[0].in_features:
            user_vec = torch.zeros(1, self.model.user_tower[0].in_features, dtype=torch.float32)
            min_dim = min(self.user_features[user_id].shape[0], self.model.user_tower[0].in_features)
            user_vec[0, :min_dim] = torch.tensor(self.user_features[user_id][:min_dim], dtype=torch.float32)
        self.model.eval()
        course_features = self.course_features
        if course_features.shape[1] != self.model.course_tower[0].in_features:
            adapted_course_features = np.zeros((len(course_features), self.model.course_tower[0].in_features))
            min_dim = min(course_features.shape[1], self.model.course_tower[0].in_features)
            adapted_course_features[:, :min_dim] = course_features[:, :min_dim]
            course_features = adapted_course_features
        predictions = np.zeros(len(self.all_course_ids))  # Заглушка
        valid_indices = [i for i, cid in enumerate(self.all_course_ids) if (cid not in exclude)]
        valid_scores = predictions[valid_indices]
        valid_courses = self.all_course_ids[valid_indices]
        top_indices = np.argsort(valid_scores)[-count:][::-1]
        return valid_courses[top_indices]
    
    def _rank_with_ensemble(self, user_id, candidate_courses):
        # Получаем индекс пользователя по его user_id (UUID или строка)
        user_id_str = str(user_id)
        if user_id_str not in self.user_id_to_idx:
            raise ValueError(f"User {user_id} not found in the database")
        
        user_idx = self.user_id_to_idx[user_id_str]
        
        # Получаем вектор пользователя по индексу
        user_vec = torch.tensor(self.user_features[user_idx], 
                            dtype=torch.float32).unsqueeze(0)
        
        # Адаптация размерности вектора пользователя
        if user_vec.shape[1] != self.model.user_tower[0].in_features:
            user_vec = torch.zeros(1, self.model.user_tower[0].in_features, dtype=torch.float32)
            min_dim = min(self.user_features[user_idx].shape[0], self.model.user_tower[0].in_features)
            user_vec[0, :min_dim] = torch.tensor(self.user_features[user_idx][:min_dim], dtype=torch.float32)
        
        self.model.eval()
        scores = []
        
        for course_id in candidate_courses:
            # Получаем индекс курса
            course_idx = np.where(self.all_course_ids == course_id)[0]
            if len(course_idx) == 0:
                continue  # Пропускаем курсы, которых нет в данных
            
            course_idx = course_idx[0]
            
            # Получаем вектор курса
            course_vec = torch.tensor(self.course_features[course_idx], 
                                    dtype=torch.float32).unsqueeze(0)
            
            # Адаптация размерности вектора курса
            if course_vec.shape[1] != self.model.course_tower[0].in_features:
                adapted_course_vec = torch.zeros(1, self.model.course_tower[0].in_features, dtype=torch.float32)
                min_dim = min(course_vec.shape[1], self.model.course_tower[0].in_features)
                adapted_course_vec[0, :min_dim] = course_vec[0, :min_dim]
                course_vec = adapted_course_vec
            
            # Вычисляем оценки
            tt_score = self.model(user_vec, course_vec).item()
            mf_score = 0.0  # Заглушка для матричной факторизации
            ensemble_score = (1 - self.mf_weight) * tt_score + self.mf_weight * mf_score
            scores.append((course_id, ensemble_score, mf_score, tt_score))
        
        # Сортируем и возвращаем результаты
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"User features: {user_vec}")
        print(f"Course features: {course_vec}")
        print(f"Predicted score: {tt_score}")
        return [[str(course_id), float(score), float(mf_score), float(tt_score)]
                for course_id, score, mf_score, tt_score in scores]


def model(app, user_id):
    with app.app_context():
        system = RecommendationSystem(app, n_recommendations=5, similarity_threshold=0.4, mf_weight=0.3)
        return system.recommend(user_id)
import numpy as np
from backend.ml_model.model.data_loading import load_data

class RecommendationSystem:
    def __init__(self, app, n_recommendations=5):
        self.users_df, self.courses_df, _ = load_data(app)
        self.n_recommendations = n_recommendations

    def recommend(self, user_id):
        # Получаем теги пользователя
        user_row = self.users_df[self.users_df['user_id'] == str(user_id)]
        if user_row.empty:
            return []
        user_tags = set(user_row.iloc[0]['preferred_tags'])
        print(user_tags)

        # Для каждого курса считаем количество совпадающих тегов
        course_scores = []
        for _, row in self.courses_df.iterrows():
            course_id = row['course_id']
            course_tags = set(row['tags'])
            score = len(user_tags & course_tags)  # число общих тегов
            course_scores.append((course_id, score))

        # Сортируем по количеству совпадающих тегов (по убыванию)
        course_scores.sort(key=lambda x: x[1], reverse=True)

        # Всегда возвращаем топ-N курсов, даже если совпадений 0
        recommendations = [course_id for course_id, score in course_scores][:self.n_recommendations]
        return recommendations


def model(app, user_id):
    system = RecommendationSystem(app)

    return system.recommend(user_id)
