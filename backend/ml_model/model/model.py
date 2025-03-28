import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class RecSysModel(nn.Module):
    def __init__(self, user_dim, course_dim, hidden_dim=64):
        super(RecSysModel, self).__init__()
        self.user_embed = nn.Linear(user_dim, hidden_dim)
        self.course_embed = nn.Linear(course_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_vec, course_vec):
        user_emb = self.user_embed(user_vec)
        course_emb = self.course_embed(course_vec)
        x = torch.cat([user_emb, course_emb], dim=1)
        return self.fc(x)

def train_model(model, train_loader, epochs=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for X_user, X_course, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_user, X_course)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model