import numpy as np
import torch
import matplotlib.pyplot as plt
from nca import CA, calc_styles, style_loss

def to_rgb(x):
    return x[...,:3,:,:]+0.5

def to_nchw(img):
    img = torch.as_tensor(img)
    if len(img.shape) == 3:
        img = img[None,...]
    return img.permute(0, 3, 1, 2)

def find_nca(style_img, size):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print("NCA", t, r-a)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    num_iterations = 100
    with torch.no_grad():
        target_style = calc_styles(to_nchw(style_img))
    ca = CA(size).cuda()
    opt = torch.optim.Adam(ca.parameters(), 1e-3)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], 0.3)
    loss_log = []
    with torch.no_grad():
        pool = ca.seed(256) # 256 CA's in the pool

    param_n = sum(p.numel() for p in ca.parameters())
    print('CA param count:', param_n)

    # training loop
    for i in range(num_iterations):
        torch.cuda.empty_cache()
        with torch.no_grad():
            batch_idx = np.random.choice(len(pool), 4, replace=False)
            x = pool[batch_idx] # x is shape (4, 12, 128, 128)
            if i%8 == 0:
                x[:1] = ca.seed(1)
        step_n = np.random.randint(32, 96)
        for k in range(step_n):
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            print(k, t, r-a)
            x = ca(x)
        imgs = to_rgb(x)
        styles = calc_styles(imgs)
        overflow_loss = (x-x.clamp(-1.0, 1.0)).abs().sum()
        loss = style_loss(styles, target_style)+overflow_loss
        print(i, overflow_loss.item(), loss.item())
        with torch.no_grad():
            loss.backward()
            for p in ca.parameters():
                p.grad /= (p.grad.norm()+1e-8)   # normalize gradients
            opt.step()
            opt.zero_grad()
            lr_sched.step()
            pool[batch_idx] = x                # update pool
    return ca
