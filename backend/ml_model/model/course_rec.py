from data_loading import load_data, prepare_features
from knn_rec import KNNRecommender
from model import RecSysModel, train_model
from data_management import add_new_user, add_interaction
import torch
from torch.utils.data import DataLoader, TensorDataset

def main():
    # Загрузка данных
    users, courses, interactions = load_data()
    user_features, course_features = prepare_features(users, courses)
    
    # Инициализация моделей
    knn_recommender = KNNRecommender(user_features)
    model = RecSysModel(user_features.shape[1], course_features.shape[1])
    
    # Подготовка данных для обучения
    user_ids = interactions["user_id"].values
    course_ids = interactions["course_id"].values
    labels = interactions["liked"].values
    
    X_users = torch.tensor(user_features[user_ids], dtype=torch.float32)
    X_courses = torch.tensor(course_features[course_ids], dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X_users, X_courses, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Обучение модели
    model = train_model(model, train_loader)
    
    # Пример использования
    def recommend_courses(user_id):
        candidate_courses = knn_recommender.get_candidate_courses(user_id, interactions)
        user_vec = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
        
        scores = []
        for course_id in candidate_courses:
            course_vec = torch.tensor(course_features[course_id], dtype=torch.float32).unsqueeze(0)
            score = model(user_vec, course_vec).item()
            scores.append((course_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [int(course_id) for course_id, _ in scores[:5]]
    
    print(recommend_courses(0))

if __name__ == "__main__":
    main()