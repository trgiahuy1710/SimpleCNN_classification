from matplotlib.testing import set_font_settings_for_testing
from networkx.classes import selfloop_edges
from tensorboard.summary.v1 import image
from torch.cuda import cudart
from torch.cuda.tunable import set_filename
from torch.xpu import device
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn
from torchvision.models.video.resnet import model_urls
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(2048, 20)
model = model.to(device)


# for name, param in model.named_parameters():
#     # if "fc." not in name:
#     #     param.requires_grad = False
#     # print(name, param.requires_grad)
#
for name, param in model.named_parameters():
    # giu ca layer 4 true
    if "fc." in name or "layer4." in name:
        pass
    else:
        param.requires_grad = False
    print(name, param.requires_grad)
summary(model, (3, 224, 224), device=str(device))
# print(model.fc)
# image = torch.randn(2, 3, 224, 224)
#
# output = model(image)
#
# print(output.shape)

class MyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.model.fc
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2= nn.Linear(1024, num_classes)

    def _forward_impl(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

if __name__ == '__main__':
    model = MyResNet()
    image = torch.rand(2, 3, 224, 224)

    out = model(image)
    print(out.shape)


    #torch.Size([2, 10])


# them vao xoa 2 phan fc layer gan them fc layer roi for ward lai