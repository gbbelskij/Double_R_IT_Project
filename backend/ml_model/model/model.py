import torch
import torch.nn as nn

INPUT_LAYER_SIZE = 38
HIDDEN_LAYER_SIZE = 60
OUTPUT_LAYER_SIZE = 80

class CourseRecommendationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._fc1 = nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(0.2)

        self._fc2 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._fc1(x)
        hidden = self._relu(hidden)
        hidden = self._dropout(hidden)
        output = self._fc2(hidden)
        output = self._softmax(output)
        return output
