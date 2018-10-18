import numpy as np
import scipy as sp
from scipy import misc
from scipy.integrate import simps
import pandas as pd
from scipy.ndimage.interpolation import geometric_transform
import os
from timeit import default_timer as timer
import tqdm


def FBT(pol, m, x_net, u_net, theta_net):
    # compute formulas 10 and 11 from article
    # Parameters:
    # pol : image resampled to polar coordinates
    # m : order of Bessel function (non integer give errors)
    # x_net : vector of x from formula 10
    # u_net, theta_net: polar grid of the given image
    # return: vector of len(x_net)

    f1 = np.exp(-1j * m * theta_net)
    f2 = pol * f1.reshape(1, -1)
    #     f2 = np.fft.fft(pol*np.exp(-1j*m), axis=1)
    fm = np.trapz(np.real(f2), theta_net, axis=1) + 1j * np.trapz(np.imag(f2), theta_net, axis=1)
    fm = fm / (2 * np.pi)

    bessel = sp.special.jv(m, u_net.reshape(-1, 1).
                           dot(x_net.reshape(1, -1)))

    ff = bessel * fm.reshape(-1, 1) * u_net.reshape(-1, 1)
    #     ff = sp.special.jn(m, u_net.dot(x_net)) * fm * u_net

    Fm = np.trapz(np.real(ff), u_net, axis=0) + 1j * np.trapz(np.imag(ff), u_net, axis=0)
    return Fm


def polar_trfm(Im, ntheta, nrad, rmax):
    # Polar Transform of the image with given numbers
    # of polar grid samples and maximal radius
    rows, cols = Im.shape
    cx = (rows + 1) / 2
    cy = (cols + 1) / 2
    #     rmax=(rows-1)/2
    #     deltatheta = 2 * np.pi/(ntheta)
    #     deltarad = rmax/(nrad-1)
    theta_int = np.linspace(0, 2 * np.pi, ntheta)
    r_int = np.linspace(0, rmax, nrad)
    theta, radius = np.meshgrid(theta_int, r_int)

    def transform(coords):
        theta1 = 2.0 * np.pi * coords[1] / ntheta
        radius1 = rmax * coords[0] / nrad
        i = cx + radius1 * np.cos(theta1)
        j = radius1 * np.sin(theta1) + cy
        return i, j

    #     xi = radius * np.cos(theta) + cx
    #     yi = radius * np.sin(theta) + cy
    PolIm = geometric_transform(Im.astype(float), transform, order=1, mode='constant', output_shape=(nrad, ntheta))
    PolIm[np.isnan(PolIm[:])] = 0
    return PolIm


# Init the bandwith B, displacement samples k

A = 64 #radius of the image
p_s = 0.5
s_ang = np.pi*A/p_s
# B = 128

# section 2.2.1: number of angular samples
# to be used in formula 10, 11

# s_ang = 2 * B
# p_s = np.pi * A / s_ang
B = s_ang/2
print('B=', B)

# section 2.2.1: number of radial samples
# to be used in formula 10, 11

s_rad = 2 * B / np.pi

# maximum expected offset of COM

ro_max = 10
# ro_max = 0.025 * 2*A #maximum expected offset of COM

# displacement samples k

k = ro_max/p_s
print('k=', k)

# k = 10
b = ro_max / 2
# p_s = 2 * b / k

# radius of the image, upper limit in formula 10

A = 2 * B * p_s / np.pi
# print("a, ps", A, p_s)

# small value to avoid duplications

eps = np.pi / (2 * k)

# bandwiths maximum abs values of m1,h1,mm

bound_m1 = np.floor(2 * b * B / A)
bound_h1 = np.floor(2 * b * B / (np.pi * A))
bound_mm = np.round(B)

Im1 = np.arange(-bound_m1, bound_m1)
Ih1 = np.arange(-bound_h1, bound_h1)
Imm = np.arange(0, bound_mm)

# Grid of angles for formula 10, 11, grid of x parameter

theta_net = np.linspace(0, 2 * np.pi, int(s_ang))
u_net = np.linspace(0, A, int(s_rad))
x_net = np.linspace(0, B / A, int(s_rad))
# print(len(theta_net), len(u_net))

# final parameters of motion

omega_net = np.linspace(-np.pi, np.pi, len(Imm))
psi_net = np.linspace(-np.pi, np.pi, 4 * b * B / A)
eta_net = np.linspace(-np.pi, np.pi, 4 * b * B / (np.pi * A))

# Check

print(len(Im1), len(psi_net))
print(len(Ih1), len(eta_net))
print(len(Imm), len(omega_net))

path_in = "/Users/anoshin_alexey/Documents/Projects/Fast-Bessel-Matching/"

im1 = misc.imread(path_in + "imm.png")

maxrad = im1.shape[0] ** 2 + im1.shape[1] ** 2
maxrad **= 0.5
maxrad = np.ceil(maxrad).astype(int)

pol1 = polar_trfm(im1, int(np.round(2 * B)), int(np.round(2 * B / np.pi)), maxrad)

print("Precount FBM of im1")
# Calculate FBT(10) for every m1,h1,mm of first image

Fm_arr = np.zeros((len(Im1) + len(Ih1) + len(Imm), len(x_net)), dtype='complex')
c2_coefs = np.zeros((len(Ih1), len(x_net)))
c1_coefs = np.zeros((len(Im1), len(x_net)))
for it_m1 in tqdm.tqdm(range(len(Im1))):
    m1 = Im1[it_m1]
    c1 = sp.special.jv(m1, b * x_net) * x_net
    c1_coefs[it_m1, :] = c1
    for it_h1 in range(len(Ih1)):
        h1 = Ih1[it_h1]
        if it_m1 == 0:
            c2 = sp.special.jv(h1, b * x_net)
            c2_coefs[it_h1, :] = c2
        for it_mm in range(len(Imm)):
            mm = Imm[it_mm]
            if Fm_arr[it_m1 + it_h1 + it_mm, :].sum() == 0:
                Fm = FBT(pol1, m1 + h1 + mm, x_net, u_net, theta_net)
                Fm_arr[it_m1 + it_h1 + it_mm, :] = Fm

im2 = misc.imread(path_in + "/test/1/imm12.png")

start = timer()

Tf = np.zeros((len(Im1), len(Ih1), len(Imm)), dtype='complex')
print(Tf.shape)
maxrad = im2.shape[0] ** 2 + im2.shape[1] ** 2
maxrad **= 0.5
maxrad = np.ceil(maxrad).astype(int)
pol2 = polar_trfm(im2, int(np.round(2 * B)), int(np.round(2 * B / np.pi)), maxrad)

# Calculate FBT(10) for every m1,h1,mm of second image

Gm_arr = np.zeros((len(Imm), len(x_net)), dtype='complex')
for it_mm in range(len(Imm)):
    mm = Imm[it_mm]
    Gm_arr[it_mm] = FBT(pol2, mm, x_net, u_net, theta_net)

# caluclate formula 25

for it_m1 in tqdm.tqdm(range(len(Im1))):
    m1 = Im1[it_m1]
    # c1 = sp.special.jn(m1, b * x_net) * x_net
    c1 = c1_coefs[it_m1, :]
    for it_h1 in range(len(Ih1)):
        h1 = Ih1[it_h1]
        c2 = c2_coefs[it_h1, :] * c1
        # c2 = sp.special.jn(h1, b * x_net) * c1
        for it_mm in range(len(Imm)):
            mm = Imm[it_mm]
            coef = 2 * np.pi * np.exp(1j * (h1 + mm) * eps)
            # Fm = FBT(pol1, m1+h1+mm, x_net, u_net, theta_net)
            #             Fm_arr[it_m1 + it_h1 + it_mm] = Fm
            Fm = Fm_arr[it_m1 + it_h1 + it_mm]
            Gm = Gm_arr[it_mm]
            func = Fm * np.conj(Gm) * c2
            Tf[it_m1, it_h1, it_mm] = np.trapz(func, x_net) - 1j * np.trapz(sp.imag(func), x_net)
            Tf[it_m1, it_h1, it_mm] *= coef

T = np.fft.ifftn(Tf)
print(T.shape)
[ipsi, ietta, iomegga] = np.unravel_index(np.argmax(T), Tf.shape)
print('Index of maximum value:', ipsi, ietta, iomegga)
psi = psi_net[ipsi]
etta = eta_net[ietta]
omegga = omega_net[iomegga]
print('Angles:')
print(np.degrees(psi), np.degrees(etta - psi), np.degrees(omegga - etta), np.degrees(eps))
print('Right values for test image:',40, 10, 5, 11.25, 40 + 10 + 5 + 11.25)
alpha = eps + omegga

rho = np.abs(np.complex(b + b * np.exp(etta - psi + eps)))
phi = np.angle(np.exp(1j * psi) * b * (1 + np.exp(1j * (etta - psi + eps))))
x = rho * np.cos(phi)
y = rho * np.sin(phi)

a = (alpha * 180 / np.pi) % 360
end = timer()
time = end - start
print(a)
# print(x, y, rho)