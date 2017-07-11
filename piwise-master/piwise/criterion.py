import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        log = F.log_softmax(outputs)
        from IPython import embed
        embed()
        exit()
        return self.loss(log, targets)