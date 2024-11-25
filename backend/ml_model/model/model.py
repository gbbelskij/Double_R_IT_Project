import torch.nn as nn

input_layer, hiden_layer, output_layer = 38, 60, 80

class CourseRecommendationModel(nn.Module):
    def __init__(self):
        super(CourseRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_layer, hiden_layer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hiden_layer, output_layer)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        output = self.softmax(output)
        return output
