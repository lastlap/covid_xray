import torch
torch.cuda.current_device()
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from pathlib import Path
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("-epochs", "--epochs", type=int, default=10, help="Enter Number of epochs")
parser.add_argument("-learning_rate", "--learning_rate", type=float, default=0.1, help="Enter Learning Rate")
args=parser.parse_args()

data_dir = 'images'

train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor()]) 

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# model = models.resnet34(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

# num_fts = model.fc.in_features

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 2),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device);

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
# loss = torch.zeros(1,requires_grad=True)
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        # loss.requires_grad = True
        # print(type(loss))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader)}")
            running_loss = 0
            model.train()

PATH = Path("./model_q1.pth")

torch.save(model,PATH)