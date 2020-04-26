import torch
torch.cuda.current_device()
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torch.autograd import Variable

parser=argparse.ArgumentParser()


parser.add_argument("-img_path", "--img_path", type=str, default=False, help="Enter Path of Image to be recognised")
args = parser.parse_args()

loader = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])
img_path= Path(args.img_path)#Path("C:/Users/abhis/Desktop/golden-retriever-puppy.jpg")

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(img_path)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

PATH = Path("./model_q1.pth")

model = torch.load(PATH)
model.eval()

logps = model.forward(image_loader(img_path))
ps = torch.exp(logps)
val=(ps==(torch.max(ps))).nonzero()[0,1]
if int(val)==0:
	print("Predicted: Covid Patient")
else:
	print("Predicted: Patient is fine")
