import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models


class DNCM(nn.Module):

    def __init__(self, k=16):
        super(DNCM, self).__init__()
        self.k = k
        self.P = nn.Parameter(torch.empty((3, self.k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((self.k, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)

    def forward(self, I, T):
        bs, _, H, W = I.shape
        I = torch.flatten(I, start_dim=2).transpose(1, 2)
        T = T.view(bs, self.k, self.k)
        # Y= DNCM(I, T) = I(h×w,3)·P(3,k)·T(k,k)·Q(k,3)
        Y = I @ self.P @ T @ self.Q
        Y = Y.view(bs, H, W, -1).permute(0, 3, 1, 2)
        return Y


class Encoder(nn.Module):
    def __init__(self, k=16):
        super(Encoder, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.requires_grad = True
        self.fc_content = nn.Linear(1000, k * k)
        self.fc_style = nn.Linear(1000, k * k)
        self.fc_content.requires_grad = True
        self.fc_style.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        d = self.fc_content(x)  # content feature
        r = self.fc_style(x)  # style feature
        return d, r

if __name__ == "__main__":
    img1 = Variable(torch.randn(4, 3, 256, 256))
    img2 = Variable(torch.randn(4, 3, 256, 256))
    encoder = Encoder()
    sDNCM = DNCM()
    nDNCM = DNCM()

    d1,r1 = encoder(img1)
    d2,r2 = encoder(img2)

    z1 = nDNCM(img1,d1)
    z2 = nDNCM(img2,d2)

    y1 = sDNCM(z2,r1)
    y2 = sDNCM(z1,r2)

    print(y1.shape,y2.shape)