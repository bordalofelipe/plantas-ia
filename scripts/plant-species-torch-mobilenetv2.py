import torch
from torch.utils.data import DataLoader, random_split
import torchvision

class PlantSpeciesModelMobileNetV2(torch.nn.Module):
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

## Setup
dataset_path = '../../inaturalist-data/data/'

batch_size = 1024
image_size = 224
num_workers = 6

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(231, 231)),
    torchvision.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.ToTensor()
])

output = {}

## Load datasets
full_dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
print('Dataset size:', len(full_dataset), 'Classes:', len(full_dataset.classes))
output['Dataset size'] = len(full_dataset)
output['Classes'] = len(full_dataset.classes)
train_dataset, val_dataset = random_split(full_dataset, [int(0.8*len(full_dataset)), len(full_dataset) - int(0.8*len(full_dataset))])
print('Using for training:', len(train_dataset))
print('Using for validation:', len(val_dataset))
output['training'] = len(train_dataset)
output['validation'] = len(val_dataset)
train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers=num_workers)

## Model
model = PlantSpeciesModelMobileNetV2(len(full_dataset.classes))

## Check for CUDA
print('Using CUDA!' if torch.cuda.is_available() else 'Not using CUDA...')
print('Using', torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
output['device'] = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

## Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

## Training
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

print('Finished training. Evaluating...')

## Validation
model.eval()
correct = 0
total = 0
running_loss = 0.0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

avg_loss = running_loss / total
accuracy = correct / total

print('avg_loss =', avg_loss)
output['avg_loss'] = avg_loss
print('accuracy =', accuracy)
output['accuracy'] = accuracy

print('Finished evaluating!')

torch.save(model, 'plant-species.pth')

import json
with open('../models/plant-species-output.json', 'w') as f:
    json.dump(output, f)
