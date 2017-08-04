import numpy as np
import torch
from blinkende_lichter.unet.network import UNet
from blinkende_lichter.unet.criterion import CrossEntropyLoss2d
from torch.optim import Adam
from torch.autograd import Variable
def IOU(output, target):
    x = np.logical_or(output==1, target==1)
    return np.sum(target * output)/sum(x[x])

def load_model(num_classes):
    net = UNet(num_classes).cuda()
    weight = torch.ones(num_classes)
    weight[0] = 0
    return net








def train_evaluate(model, epoch_num, training_loader, validation, patience, maxchecks):
    best_score = None
    model.train()
    criterion = CrossEntropyLoss2d()
    optimizer = Adam(model.parameters())
    epoch_loss = []
    val_epoch_loss = []
    #train_inputs = []
    train_targets = []
    train_outputs = []
    #val_inputs = []
    val_targets = []
    val_outputs = []
    for epoch in range(1, epoch_num):
        for step, (images, labels) in enumerate(training_loader):
            images = images.cuda()
            labels = labels.cuda()
            inputs = Variable(images)
            targets = Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss.append((loss.data[0]))


        model.eval()

        for i in range(len(validation.data[0])):
            input = torch.FloatTensor(validation.data[0][i]).cuda()
            target = Variable(torch.from_numpy(validation.data[1][i])).cuda()
            output = model(Variable(input).unsqueeze(0))
            x = np.exp(output.data.cpu().numpy())
            res = x/x.sum(1)
            pos = np.argmax(res, axis=1).squeeze()
            val_loss = IOU(pos, target.data.cpu().numpy())
            val_epoch_loss.append(val_loss)
            if(epoch>=patience):
                if(len(val_epoch_loss) % 4 == 0):
                    score = sum(val_epoch_loss[-4:])/len(val_epoch_loss[-4:])            
                    if(best_score == None):
                        best_score = score
                    if(score >= best_score):
                        best_score = score
                        checks = 0
                    else:
                        checks+=1
                    if (checks>=maxchecks):
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
    print("training and evaluation complete")
    return val_epoch_loss, epoch_loss, best_score