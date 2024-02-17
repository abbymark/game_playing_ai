import torch

# class CNNActor(torch.nn.Module):
#     def __init__(self, input_channels, output_dim):
#         super().__init__()
#         self.input_channels = input_channels
#         self.output_dim = output_dim
#         self.conv1 = torch.nn.Conv2d(input_channels, 16, 3, 1, 1)
#         self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
#         self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 1)

#         self.max_pool = torch.nn.MaxPool2d(2, 2)
#         self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.output_layer = torch.nn.Linear(64, output_dim)
#         self.relu = torch.nn.ReLU()

#         self.softmax = torch.nn.Softmax(dim=-1)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.global_avg_pool(x)
#         x = x.view(-1, 64)
#         x = self.output_layer(x)
#         x = self.softmax(x)
#         return x

class CNNActor(torch.nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.actor = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, output_dim),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.actor(x)


    
# class CNNCritic(torch.nn.Module):
#     def __init__(self, input_channels):
#         super().__init__()
#         self.input_channels = input_channels
#         self.conv1 = torch.nn.Conv2d(input_channels, 16, 3, 1, 1)
#         self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
#         self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 1)

#         self.max_pool = torch.nn.MaxPool2d(2, 2)
#         self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

#         self.relu = torch.nn.ReLU()

#         self.fc1 = torch.nn.Linear(64, 1)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.global_avg_pool(x)
#         x = x.view(-1, 64)
#         x = self.fc1(x)
#         return x
    
class CNNCritic(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.critics = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 16, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.critics(x)