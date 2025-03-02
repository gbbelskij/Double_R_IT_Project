# model.py
import torch
import torch.nn as nn

class CourseRecommender(nn.Module):
    def __init__(self, num_tags, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_tags, embedding_dim)
        
    def forward(self, courses, users):
        course_emb = self.embedding(courses).mean(dim=1)  # [num_courses, emb]
        user_emb = self.embedding(users).mean(dim=1)      # [num_users, emb]
        similarity = torch.mm(course_emb, user_emb.T)     # [num_courses, num_users]
        return similarity