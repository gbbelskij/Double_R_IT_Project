import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, course_dim, embedding_dim=64, tower_hidden_dims=[128, 64]):
        """
        Two Tower model for recommendation system
        
        Args:
            user_dim: dimension of user features
            course_dim: dimension of course features
            embedding_dim: final embedding dimension for both towers
            tower_hidden_dims: hidden dimensions for tower layers
        """
        super(TwoTowerModel, self).__init__()
        
        # User tower
        user_layers = []
        prev_dim = user_dim
        for hidden_dim in tower_hidden_dims:
            user_layers.append(nn.Linear(prev_dim, hidden_dim))
            user_layers.append(nn.ReLU())
            user_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        user_layers.append(nn.Linear(prev_dim, embedding_dim))
        user_layers.append(nn.BatchNorm1d(embedding_dim))
        
        # Course tower
        course_layers = []
        prev_dim = course_dim
        for hidden_dim in tower_hidden_dims:
            course_layers.append(nn.Linear(prev_dim, hidden_dim))
            course_layers.append(nn.ReLU())
            course_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        course_layers.append(nn.Linear(prev_dim, embedding_dim))
        course_layers.append(nn.BatchNorm1d(embedding_dim))
        
        # Define the towers as Sequential models
        self.user_tower = nn.Sequential(*user_layers)
        self.course_tower = nn.Sequential(*course_layers)
        
        print("=" * 50)
        print("Initializing Two Tower model...")
        print(f"- User feature dimension: {user_dim}")
        print(f"- Course feature dimension: {course_dim}")
        print(f"- Embedding dimension: {embedding_dim}")
        print(f"- Tower hidden dimensions: {tower_hidden_dims}")
        print("=" * 50)
    
    def forward(self, user_vec, course_vec):
        """
        Forward pass through the network
        
        Args:
            user_vec: user feature vector
            course_vec: course feature vector
            
        Returns:
            Predicted probability of user liking the course
        """
        # Get embeddings from towers
        user_embedding = self.user_tower(user_vec)
        course_embedding = self.course_tower(course_vec)
        
        # Compute dot product between embeddings
        # Normalize embeddings first for cosine similarity
        user_embedding = nn.functional.normalize(user_embedding, p=2, dim=1)
        course_embedding = nn.functional.normalize(course_embedding, p=2, dim=1)
        
        # Dot product for similarity score
        dot_product = torch.sum(user_embedding * course_embedding, dim=1, keepdim=True)
        
        # Scale to [0, 1] for probability interpretation
        prediction = torch.sigmoid(dot_product)
        
        return prediction
    
    def get_user_embedding(self, user_vec):
        """Extract user embedding vector"""
        with torch.no_grad():
            return self.user_tower(user_vec)
    
    def get_course_embedding(self, course_vec):
        """Extract course embedding vector"""
        with torch.no_grad():
            return self.course_tower(course_vec)
    
    def predict_all_courses(self, user_vec, course_features):
        """
        Predict scores for one user and multiple courses
        
        Args:
            user_vec: single user feature vector (1, user_dim)
            course_features: features for all courses (n_courses, course_dim)
            
        Returns:
            Array of prediction scores
        """
        course_tensor = torch.tensor(course_features, dtype=torch.float32)
        
        # Get user embedding once (no need to repeat)
        with torch.no_grad():
            user_embedding = self.user_tower(user_vec)
            user_embedding = nn.functional.normalize(user_embedding, p=2, dim=1)
            
            # Process all course features
            course_embeddings = self.course_tower(course_tensor)
            course_embeddings = nn.functional.normalize(course_embeddings, p=2, dim=1)
            
            # Calculate similarity for all courses at once
            # Expand user embedding for matrix multiplication
            similarity = torch.matmul(user_embedding, course_embeddings.t()).squeeze()
            predictions = torch.sigmoid(similarity)
            
        return predictions.numpy()


def train_model(model, train_loader, epochs=1000, lr=0.001, weight_decay=1e-5):
    """
    Train the two tower model
    
    Args:
        model: TwoTowerModel instance
        train_loader: DataLoader with training data
        epochs: number of training epochs
        lr: learning rate
        weight_decay: L2 regularization strength
    
    Returns:
        Trained model
    """
    print("=" * 50)
    print(f"Training Two Tower model for {epochs} epochs...")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        
        for X_user, X_course, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_user, X_course)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/batches:.4f}")
    
    print("Training completed!")
    print("=" * 50)
    return model