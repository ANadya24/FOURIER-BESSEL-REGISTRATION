import numpy as np
import scipy as sp
from scipy import misc
from scipy import ndimage
import os
import random

random.seed(1234)

im1 = misc.imread("im1.png")

os.makedirs("dataset/1", exist_ok=True)
N = 10
t = np.array([3, 4])
print("abs:",np.abs(t))
for i in range(N):
    im2 = ndimage.interpolation.shift(im1, t)
    im22 = ndimage.interpolation.rotate(im2, 360 * i / N, reshape=False)
    strl = ['dataset/1/im_3_4_', int(360 * i / N), '.png']
    sp.misc.imsave(''.join(map(str, strl)), im22)

os.makedirs("dataset/2", exist_ok=True)
a = 30
print("angle:", a)
for i in range(N):
    t = random.sample(range(8), 2)
    im2 = ndimage.interpolation.shift(im1, t)
    im22 = ndimage.interpolation.rotate(im2, a, reshape=False)
    misc.imsave(''.join(map(str, ['dataset/2/im_', t[0], '_', t[1], '_30.png'])), im22)



