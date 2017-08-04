import numpy as np
from torch.utils.data import DataLoader
from attorch.dataset import ListDataset
from blinkende_lichter.datatools import RandomFlip, TypeConversion
import torchvision.transforms as transforms
def process(Filename):
    transform = transforms.Compose([    
        RandomFlip(),    
        TypeConversion(np.float32, np.long) ]) 
    dat = np.load(Filename)

    tr = dat['train']
    vd = dat['validation']
    for image in tr[1]:
        for h in range(len(image)):
            for w in range(len(image[0])):
                if(image[h][w]==0):
                    image[h][w]=0
                else:
                    image[h][w] = 1


    for image in vd[1]:
        for h in range(len(image)):
            for w in range(len(image[0])):
                if(image[h][w]==0):
                    image[h][w]=0
                else:
                    image[h][w] = 1





    train = ListDataset(*tr, transform=transform)

    val = ListDataset(*vd) 
    trainloader = DataLoader( train, shuffle=True, batch_size=1)
    
    return train, val, trainloader