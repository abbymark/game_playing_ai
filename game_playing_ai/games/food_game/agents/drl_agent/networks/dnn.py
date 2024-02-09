import torch

# class DNN(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.fc1 = torch.nn.Linear(input_dim, 128)
#         self.fc2 = torch.nn.Linear(128, 256)
#         self.fc3 = torch.nn.Linear(256, 64)
#         self.fc4 = torch.nn.Linear(64, output_dim)
#         self.relu = torch.nn.ReLU()
    
#     def forward(self, x):
#         x = x.view(-1, self.input_dim)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.fc4(x)
#         return x

class DNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    