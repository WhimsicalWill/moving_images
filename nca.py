import numpy as np
import matplotlib.pylab as pl
import torch
import torchvision.models as models

# VGG16-based Style Model
torch.set_default_tensor_type('torch.cuda.FloatTensor')
vgg16 = models.vgg16(pretrained=True).features

def calc_styles(imgs):
    style_layers = [1, 6, 11, 18, 25]
    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
    std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
    x = ((imgs-mean) / std)
    grams = []
    for i, layer in enumerate(vgg16[:max(style_layers)+1]):
        x = layer(x)
        if i in style_layers:
            h, w = x.shape[-2:]
            y = x.clone()  # workaround for pytorch in-place modification bug(?)
            gram = torch.einsum('bchw, bdhw -> bcd', y, y) / (h*w)
            grams.append(gram)
    return grams

def style_loss(grams_x, grams_y):
    loss = 0.0
    for x, y in zip(grams_x, grams_y):
        loss = loss + (x-y).square().mean()
    return loss

class CA(torch.nn.Module):

    def __init__(self, size=128, chn=12, hidden_n=96):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = size
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn*4, hidden_n, 1)
        self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()
        self.ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
        self.sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
        self.lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])

    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b*ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:,None])
        return y.reshape(b, -1, h, w)

    def perception(self, x):
        filters = torch.stack([self.ident, self.sobel_x, self.sobel_x.T, self.lap])
        return self.perchannel_conv(x, filters)

    def forward(self, x, update_rate=0.5):
        y = self.perception(x)
        y = self.w2(torch.relu(self.w1(y)))
        b, c, h, w = y.shape
        udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
        return x+y*udpate_mask

    def seed(self, n, sz=None):
        if sz is None:
            sz = self.size
        return torch.zeros(n, self.chn, sz, sz)
