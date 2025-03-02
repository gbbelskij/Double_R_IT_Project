import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np

class RecommendationModel(nn.Module):
    def __init__(self, num_tags, embedding_dim):
        super(RecommendationModel, self).__init__()
        self.tag_embedding = nn.Embedding(num_tags, embedding_dim)
        self.flatten = nn.Flatten()

    def forward(self, course_tags, user_interests):
        # Преобразуем теги в embedding'и
        course_embedding = self.tag_embedding(course_tags)
        user_embedding = self.tag_embedding(user_interests)

        # Усредняем embedding'и по тегам
        course_embedding = torch.mean(course_embedding, dim=1)
        user_embedding = torch.mean(user_embedding, dim=1)

        # Вычисляем косинусное сходство
        similarity = torch.sum(course_embedding * user_embedding, dim=1, keepdim=True)
        return similarity