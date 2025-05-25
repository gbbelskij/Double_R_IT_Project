from backend.ml_model.model.data_loading import load_data
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
from backend.database.User import Course
import uuid

class RecommendationSystem:
    def __init__(self, app, n_recommendations=5):
        with app.app_context():
            self.users, self.courses, self.interactions = load_data(app)
            self.dataset = Dataset()
            print(self.users)
            self.dataset.fit(
                self.users['user_id'].unique(),
                self.courses['course_id'].unique()
            )
            self.interactions_matrix, _ = self.dataset.build_interactions(
                (row['user_id'], row['course_id'], row['liked']) 
                for idx, row in self.interactions.iterrows()
            )
            self.model = LightFM(loss='warp')
            self.model.fit(self.interactions_matrix, epochs=30)
            self.n_recommendations = n_recommendations

    def recommend(self, user_id):
        user_internal_id = self.dataset.mapping()[0][user_id]
        scores = self.model.predict(user_internal_id, np.arange(self.interactions_matrix.shape[1]))
        top_courses = np.argsort(-scores)
        course_id_map = {v: k for k, v in self.dataset.mapping()[2].items()}
        # names = []
        # for i in top_courses[:self.n_recommendations]:
        #     # Преобразуем строку/UUID в объект UUID, если нужно
        #     course_obj = Course.query.filter_by(course_id=uuid.UUID(str(course_id_map[i]))).first()
        #     if course_obj:
        #         names.append(course_obj.title)  # или любой другой нужный атрибут
        # print(names)
        return [course_id_map[i] for i in top_courses[:self.n_recommendations]]

def model(app, user_id):
    with app.app_context():
        system = RecommendationSystem(app, n_recommendations=5)
        return(system.recommend(user_id))
