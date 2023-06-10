import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class BinaryCrossEntropyLoss(CrossEntropyLoss):
    def forward(self, input, target):
        input1, input2 = input
        loss1 = F.cross_entropy(input1, target, weight=self.weight,
                                ignore_index=self.ignore_index, reduction=self.reduction,
                                label_smoothing=self.label_smoothing)
        loss2 = F.cross_entropy(input2, target, weight=self.weight,
                                ignore_index=self.ignore_index, reduction=self.reduction,
                                label_smoothing=self.label_smoothing)
        loss = loss1 + loss2

        return loss
