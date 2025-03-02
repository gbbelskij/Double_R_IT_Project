# model.py
import torch
import torch.nn as nn

class CourseRecommender(nn.Module):
    def __init__(self, num_tags, embedding_dim=16, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(num_tags, embedding_dim)

        # Self-Attention слой
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # Полносвязный слой для подсчета финального сходства
        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, courses, users):
        course_emb = self.embedding(courses)  # (batch, seq_len, emb_dim)
        user_emb = self.embedding(users)      # (batch, seq_len, emb_dim)

        # Attention: пытаемся сфокусироваться на самых важных признаках
        course_emb, _ = self.attention(course_emb, course_emb, course_emb)
        user_emb, _ = self.attention(user_emb, user_emb, user_emb)

        # Усредняем по тегам, чтобы получить один вектор на курс/пользователя
        course_emb = course_emb.mean(dim=1)  # (batch, emb_dim)
        user_emb = user_emb.mean(dim=1)      # (batch, emb_dim)

        # Считаем сходство через скалярное произведение
        similarity = torch.mm(course_emb, user_emb.T)

        return similarity
