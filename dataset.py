import numpy as np
import scipy as sp
from scipy import misc
from scipy import ndimage
import os
import random

random.seed(1234)

im1 = misc.imread("im1.png")

os.makedirs("dataset/3", exist_ok=True)
N = 10
t = 5
print("abs:",np.abs(t))
for i in range(N):
    a = 2 * np.pi * i / N
    transl = np.array([np.real(t*np.exp(1j*a)), np.imag(t*np.exp(1j*a))])
    print(transl)
    im2 = ndimage.interpolation.shift(im1, transl)
    # im22 = ndimage.interpolation.rotate(im2, 360 * i / N, reshape=False)
    strl = ['dataset/3/im_', t, '_', int(360 * i / N), '.png']
    sp.misc.imsave(''.join(map(str, strl)), im2)

os.makedirs("dataset/4", exist_ok=True)
a = np.pi/6
aa = 30
print("angle:", a)
for i in range(N):
    # t = random.sample(range(8), 2)
    # t = np.random.randint(1, 10)
    t = i+1
    transl = np.array([np.real(t * np.exp(1j * a)), np.imag(t * np.exp(1j * a))])
    print(transl)
    im2 = ndimage.interpolation.shift(im1, transl)
    # im22 = ndimage.interpolation.rotate(im2, a, reshape=False)
    misc.imsave(''.join(map(str, ['dataset/4/im_', t, '_', aa, '.png'])), im2)



