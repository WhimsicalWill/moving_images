import torch
import numpy as np
from moving_image import find_nca, to_rgb
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import imageio

max_size = 64
test_img = Image.open("leaves.jpg")
test_img.thumbnail((max_size, max_size), Image.ANTIALIAS)
test_img = np.float32(np.asarray(test_img)) / 255.0
print(test_img.shape)
ca = find_nca(test_img, 128)

with torch.no_grad():
    out_size = 64
    num_frames = 60
    vid = []
    x = ca.seed(1, out_size)

    writer = imageio.get_writer('some_file.mp4', fps=60)

    for k in range(num_frames):
        step_n = min(2**(k//30), 16) # speeds up until constant at k=120
        for i in range(step_n):
            x[:] = ca(x)
        # img = to_rgb(x[0]).permute(1, 2, 0).cpu()
        img = to_rgb(x[0]).permute(1, 2, 0).cpu()
        print(torch.min(img), torch.max(img), img.shape)
        img = torch.clamp(img, 0, 1) * 255
        writer.append_data(np.uint8(np.asarray(img)))
    writer.close()
    # before converting, make sure that the values are correct in the tensor
    # wrap any overflow back into the appropriate range

    # vid.add(zoom(img, 2))

    # brennan, ryan
    # eli, lainie
    # jeremy, anne
