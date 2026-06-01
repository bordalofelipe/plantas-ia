import torch
import torchvision

class PlantDiseaseModelMobileNetV2(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
        self.transform = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.model = torchvision.models.mobilenet_v2(weights=weights)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        # Freeze backbone
        for param in self.model.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)

class PlantDiseaseModelResNet50(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.transform = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.model = torchvision.models.resnet50(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        # Freeze backbone
        #for name, param in self.model.named_parameters():
        #    if not name.startswith('fc.'):
        #        param.requires_grad = False

    def forward(self, x):
        x = self.transform(x)
        return self.model(x)

model = torch.load('../models/plant-disease.pth', weights_only = False)

print(type(model))

# After loading your PyTorch model
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
dummy_input = dummy_input.to(torch.device('cuda'))
torch.onnx.export(model, dummy_input, "../docs/plant-disease.onnx", 
    input_names=['input.1'],
    output_names=['output']
)
