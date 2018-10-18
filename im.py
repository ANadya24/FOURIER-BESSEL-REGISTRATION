import numpy as np
import scipy as sp
from scipy import misc
from scipy import ndimage
import os
import random
import pandas as pd

cols = ['name', 'b', 'ksi', 'etta', 'omega', 'eps']
df = pd.DataFrame(columns=cols)
ksii = [20,40,60]
ettaa = [10,33]
omegaa = [5,25]
bb = [3,4,5]
random.seed(1234)
os.makedirs("test/2", exist_ok=True)

im1 = misc.imread("imm.png")
A = 64 #radius of the image
p_s = 1
s_ang = np.pi*A/p_s
B = s_ang/2
s_rad = 2*B/np.pi
i = 0
for ksi in ksii:
    for etta in ettaa:
        for w in omegaa:
            for b in bb:
                ro_max = 2*b
                # ro_max = 0.025 * 2*A #maximum expected offset of COM
                k = ro_max/p_s
                eps = np.pi/(2*k)
                # print(ro_max, k,eps)
                eps = 360 * eps * 0.5 / np.pi
                # print(ro_max, k,eps)

                # w = 30
                # b = 4
                # etta = 45
                # ksi = 20
                psi2 = w
                rho2 = b
                fi2 = etta + eps
                # print(fi2)
                psi1 = 0
                rho1 = b
                fi1 = ksi
                im2 = ndimage.interpolation.rotate(im1, psi2, reshape=False)
                im2 = ndimage.interpolation.shift(im2, [0,-rho2])
                im2 = ndimage.interpolation.rotate(im2, fi2, reshape=False)

                im2 = ndimage.interpolation.rotate(im2, psi1, reshape=False)
                im2 = ndimage.interpolation.shift(im2, [0, -rho1])
                im2 = ndimage.interpolation.rotate(im2, fi1, reshape=False)
                strl = ['imm' ,i, '.png']
                strr = ['test/1/imm' ,i, '.png']
                sp.misc.imsave(''.join(map(str, strr)), im2)
                towrt = {'name': strl, 'b': b, 'ksi': ksi, 'etta': etta, 'omega': w, 'eps': eps}
                df = df.append(towrt, ignore_index=True)
                i += 1
df.to_csv('test/2.csv', columns=cols)
