from torch import nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, classes=10):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool2d((32,32))
        )
        
        self.conv2 = nn.Sequential(
            
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            nn.AdaptiveMaxPool2d((8,8))
        )  

        self.dense = nn.Sequential(
            
            nn.Linear(512*8*8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, classes),
            nn.BatchNorm1d(classes),
            nn.LogSoftmax(dim=1)
        )

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=1)
        x = self.dense(x)
        return x

    
class CNNTransfer(nn.Module):
    def __init__(self, classes=10):
        super(CNNTransfer, self).__init__()
        
        # Load pre-trained EfficientNet and freeze all layers
        self.efficientnet = models.efficientnet_v2_s(weights='DEFAULT')
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        # Replace the classifier with custom layers
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, classes),
            nn.BatchNorm1d(classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.efficientnet(x)
        return x