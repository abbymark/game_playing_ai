import torch

class CNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.output_layer = torch.nn.Linear(64, output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # add a new dimension to the tensor
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 64)
        x = self.output_layer(x)
        return x

# import torch

# class CNN(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
#         self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

#         self.output_layer = torch.nn.Linear(512, output_dim)
#         self.relu = torch.nn.ReLU()
    
#     def forward(self, x):
#         # add a new dimension to the tensor
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv4(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv5(x)
#         x = self.relu(x)
#         x = self.max_pool(x)
#         x = self.conv6(x)
#         x = self.relu(x)
#         x = self.global_avg_pool(x)
#         x = x.view(-1, 512)
#         x = self.output_layer(x)
#         return x
    