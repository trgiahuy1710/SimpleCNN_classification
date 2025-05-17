import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 32, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )


    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"), #same tự tính toán padding để in = output
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'), # valid = 0
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2) # pooling là để giảm size,
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # shape B x C x 7 x 7 co padding = same
        # => khong có padding B x C x 18 x 18
        x = self.conv5(x)
        # shape B x C x 7 x 7

        # shapp bay gio la 7 x 7 x 128
        #flatten
        x = x.view(x.shape[0], -1)
        # conv giam xuong con 7x7x128 di qua 5 conv co 128 kenh de extract feature - với padding ở đây là same - 1

        # flatten thanh 6272 = 7 x 7 x 128 roi truyen vao fc layer
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # model = SimpleNeuralNetwork()
    model = SimpleCNN()
    input_data = torch.rand(8, 3, 224, 224)
    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()
    while True:
        result = model(input_data)
        print(result.shape)
        break
