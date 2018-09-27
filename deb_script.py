import numpy as np
import scipy as sp
from scipy import misc
import pandas as pd
from scipy.ndimage.interpolation import geometric_transform
import os
from timeit import default_timer as timer

def FBT(pol, m, x_net, u_net, theta_net):
    f1 = np.exp(-1j*m*theta_net)
    f2 = pol * f1
    fm = np.trapz(f2, theta_net, axis=1)
    fm = fm / (2 * np.pi)
    Fm = np.zeros(x_net.shape, dtype=fm.dtype)
#     flag = m < 0
    ff = sp.special.jn(abs(m), u_net.reshape(-1,1).
                       dot(x_net.reshape(1,-1))) * fm.reshape(-1,1) * u_net.reshape(-1,1)
    Fm = np.trapz(np.real(ff), u_net, axis=0) + 1j*np.trapz(np.imag(ff), u_net, axis=0)
    return Fm

def polar_trfm(Im, ntheta, nrad, rmax):
    #Polar Transform
    rows, cols = Im.shape
    cx = (rows+1)/2
    cy = (cols+1)/2
#     rmax=(rows-1)/2
#     deltatheta = 2 * np.pi/(ntheta)
#     deltarad = rmax/(nrad-1)
    theta_int = np.linspace(0, 2*np.pi, ntheta)
    r_int = np.linspace(0, rmax, nrad)
    theta, radius = np.meshgrid(theta_int, r_int)
    def transform(coords):
        theta = 2.0*np.pi*coords[1] / ntheta
        radius = rmax * coords[0] / nrad
        i = cx + radius*np.cos(theta)
        j = radius*np.sin(theta) + cy
        return i,j
#     xi = radius * np.cos(theta) + cx
#     yi = radius * np.sin(theta) + cy
    PolIm = geometric_transform(Im.astype(float), transform, order=1, mode='constant', output_shape=(nrad, ntheta))
    PolIm[np.isnan(PolIm[:])] = 0
    return PolIm

A = 80 #radius of the image
p_s = 1
s_ang = np.pi*A/p_s
B = s_ang/2
s_rad = 2*B/np.pi
# ro_max = 7
ro_max = 0.025 * 2*A #maximum expected offset of COM
k = ro_max/p_s
b = ro_max/2
eps = np.pi/(2*k)
# Imm = np.linspace(-B, B, 5)
bound_m1 = np.floor(2*b*B/A)
bound_h1 = np.floor(2*b*B/(np.pi * A))
bound_mm = np.floor(B)
Im1 = np.arange(-bound_m1, bound_m1)
Ih1 = np.arange(-bound_h1, bound_h1)
Imm = np.arange(0, bound_mm)

theta_net = np.linspace(0, 2*np.pi, int(s_ang))
u_net = np.linspace(0, A, int(s_rad))
x_net = np.arange(0, B/A, np.pi/(2*A))
omega_net = np.linspace(0, 2*np.pi, len(Imm))
psi_net = np.linspace(0, 2*np.pi, int(np.pi*k))
eta_net = np.linspace(0, 2*np.pi, int(k))

path_in = "/Users/anoshin_alexey/Documents/Projects/Fast-Bessel-Matching/"

im1 = misc.imread(path_in + "im1.png")
pol1 = polar_trfm(im1, int(s_ang), int(s_rad), A)

print("Precount FBM of im1")

Fm_arr = np.zeros((len(Im1) + len(Ih1) + len(Imm), len(x_net)), dtype='complex')
# c2_coefs = np.zeros((len(Im1), len(Ih1), len(x_net)))
for it_m1 in range(len(Im1)):
    m1 = Im1[it_m1]
    # c1 = sp.special.jn(m1, b * x_net) * x_net
    for it_h1 in range(len(Ih1)):
        h1 = Ih1[it_h1]
        # c2 = sp.special.jn(h1, b * x_net) * c1
        # c2_coefs[it_m1, it_h1, :] = c2
        for it_mm in range(len(Imm)):
            mm = Imm[it_mm]
            if Fm_arr[it_m1 + it_h1 + it_mm, :].sum() == 0:
                Fm = FBT(pol1, m1+h1+mm, x_net, u_net, theta_net)
                Fm_arr[it_m1 + it_h1 + it_mm, :] = Fm

cols = ['GT_rho', 'GT_x', 'GT_y', 'x', 'y', 'rho', 'psi', 'etta', 'omega', 'time']
df = pd.DataFrame(columns=cols)
path = path_in + "dataset/3/"
for f in os.listdir(path):
    if f.endswith('.png'):
        im2 = misc.imread(os.path.join(path, f))
    else:
        continue
    start = timer()
    Tf = np.zeros((len(Im1), len(Ih1), len(Imm)), dtype='complex')
    pol2 = polar_trfm(im2, int(s_ang), int(s_rad), A)

    for it_m1 in range(len(Im1)):
        m1 = Im1[it_m1]
        c1 = sp.special.jn(m1, b * x_net) * x_net
        for it_h1 in range(len(Ih1)):
            h1 = Ih1[it_h1]
            # c2 = c2_coefs[it_m1, it_h1, :]
            c2 = sp.special.jn(h1, b * x_net) * c1
            for it_mm in range(len(Imm)):
                mm = Imm[it_mm]
                coef = 2*np.pi * np.exp(1j*(h1+mm)*eps)
                # Fm = FBT(pol1, m1+h1+mm, x_net, u_net, theta_net)
    #             Fm_arr[it_m1 + it_h1 + it_mm] = Fm
                Fm = Fm_arr[it_m1 + it_h1 + it_mm]
                Gm = FBT(pol2, mm, x_net, u_net, theta_net)
                func = Fm*np.conj(Gm)*c2
                Tf[it_m1, it_h1, it_mm] = np.trapz(sp.real(func), x_net) \
                                          + 1j*np.trapz(sp.imag(func), x_net)
                Tf[it_m1, it_h1, it_mm] = Tf[it_m1, it_h1, it_mm] * coef

    T = np.fft.ifftn(Tf)
    # print(T.shape)
    [ipsi, ietta, iomegga] = np.unravel_index(np.argmax(T, axis=None), T.shape)
    psi = psi_net[ipsi]
    etta = eta_net[ietta]
    omegga = omega_net[iomegga]
    # print(psi, etta, omegga)
    alpha = eps + omegga
    phi = np.angle(np.exp(1j*psi)*b*(1 + np.exp(1j*(etta - psi + eps))))
    rho = np.abs(b * np.sqrt(2*(1 + np.cos(etta - psi + eps))))
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    a = alpha * 180 / np.pi - 360
    end = timer()
    time = end - start
    print(f)
    st = f.split('_')
    gt_rho = st[1]
    gt_a = st[2].split('.')[0]
    gt_x = np.int(gt_rho) * np.cos(np.radians(np.int(gt_a)))
    gt_y = np.int(gt_rho) * np.sin(np.radians(np.int(gt_a)))
    towrt = {'GT_rho': gt_rho, 'GT_x': gt_x, 'GT_y': gt_y, 'x': x,
             'y': y, 'rho': rho, 'psi': psi, 'etta': etta, 'omega': omegga, 'time': time}
    df = df.append(towrt, ignore_index=True)
df.to_csv(path_in + 'dataset/3.csv', columns=cols)

print("Done first")

df = pd.DataFrame(columns=cols)
path = path_in + "dataset/4/"
for f in os.listdir(path):
    start = timer()
    if f.endswith('.png'):
        im2 = misc.imread(os.path.join(path, f))
    else:
        continue
    Tf = np.zeros((len(Im1), len(Ih1), len(Imm)), dtype='complex')
    pol2 = polar_trfm(im2, int(s_ang), int(s_rad), A)

    for it_m1 in range(len(Im1)):
        m1 = Im1[it_m1]
        c1 = sp.special.jn(m1, b * x_net) * x_net
        for it_h1 in range(len(Ih1)):
            h1 = Ih1[it_h1]
            # c2 = c2_coefs[it_m1, it_h1, :]
            c2 = sp.special.jn(h1, b * x_net) * c1
            for it_mm in range(len(Imm)):
                mm = Imm[it_mm]
                coef = 2*np.pi * np.exp(1j*(h1+mm)*eps)
                # Fm = FBT(pol1, m1+h1+mm, x_net, u_net, theta_net)
    #             Fm_arr[it_m1 + it_h1 + it_mm] = Fm
                Fm = Fm_arr[it_m1 + it_h1 + it_mm]
                Gm = FBT(pol2, mm, x_net, u_net, theta_net)
                func = Fm*np.conj(Gm)*c2
                Tf[it_m1, it_h1, it_mm] = np.trapz(sp.real(func), x_net) \
                                          + 1j*np.trapz(sp.imag(func), x_net)
                Tf[it_m1, it_h1, it_mm] = Tf[it_m1, it_h1, it_mm] * coef

    T = np.fft.ifftn(Tf)
    # print(T.shape)
    [ipsi, ietta, iomegga] = np.unravel_index(np.argmax(T, axis=None), T.shape)
    psi = psi_net[ipsi]
    etta = eta_net[ietta]
    omegga = omega_net[iomegga]
    # print(psi, etta, omegga)
    alpha = eps + omegga
    phi = np.angle(np.exp(1j*psi)*b*(1 + np.exp(1j*(etta - psi + eps))))
    rho = np.abs(b * np.sqrt(2*(1 + np.cos(etta - psi + eps))))
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    a = alpha * 180 / np.pi - 360
    end = timer()
    time = end - start
    print(f)
    # st = f.split('_')
    # gt_x = st[1]
    # gt_y = st[2]
    # gt_a = st[3].split('.')[0]
    # towrt = {'GT_angle': gt_a, 'GT_x': gt_x, 'GT_y': gt_y, 'angle': a, 'x': x,
    #          'y': y, 'psi': psi, 'etta': etta, 'omega': omegga, 'time': time}
    st = f.split('_')
    gt_rho = st[1]
    gt_a = st[2].split('.')[0]
    gt_x = np.int(gt_rho) * np.cos(np.radians(np.int(gt_a)))
    gt_y = np.int(gt_rho) * np.sin(np.radians(np.int(gt_a)))
    towrt = {'GT_rho': gt_rho, 'GT_x': gt_x, 'GT_y': gt_y, 'x': x,
             'y': y, 'rho': rho, 'psi': psi, 'etta': etta, 'omega': omegga, 'time': time}
    df = df.append(towrt, ignore_index=True)
df.to_csv(path_in + 'dataset/4.csv', columns=cols)

print("Done second")
