from typing import List, Optional, Dict, Any
import numpy as np
from skimage.measure import label, regionprops
import cv2
import scipy as sp
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import time

from utils import polar_trfm, normalize_alpha
from matrix_utils import (
    get_translation_mat,
    get_mat_2x3,
    get_rotation_mat, mat_inv,
    warp_img,
    show_img_ovl
)
from fourier_bessel_transform import FBT, FBT_Laguerre, FBT_Laguerre_fast
from laguerre import Laguerre


def shift(im, vec):
    mat_trans = get_translation_mat(*vec)
    return cv2.warpAffine(im, get_mat_2x3(mat_trans), 
                          (im.shape[1], im.shape[0]))
    
    
def rotate(im, angle, radians=False, center=None):
    h, w = im.shape[:2]
    if center is None:
        center = np.array([w // 2, h // 2])
    mat_trans_minus_center = get_translation_mat(-center[0], -center[1])
    mat_rot = get_rotation_mat(angle, radians)
    mat_trans_center = get_translation_mat(*center)
    return cv2.warpAffine(im, get_mat_2x3(mat_trans_center @ mat_rot @ mat_trans_minus_center),
                          (im.shape[1], im.shape[0]))


def set_integration_intervals(image_radius: int = 128,
                              pixel_sampling: float = 1., rho_max: float = 7.):
    '''
    Init the bandwith B, displacement samples k
    :param image_radius: radius of the image
    :param pixel_sampling:
    :param rho_max: maximum expected offset of COM (0.025 * image_diameter)
    :return:
    '''

    # image_radius = 2 * bandwidth * pixel_sampling / np.pi
    # image_radius = 128

    # section 2.2.1: number of angular samples
    # to be used in formula 10, 11
    s_ang = np.pi * image_radius / pixel_sampling

    # section 2.2.1: number of radial samples
    # to be used in formula 10, 11
    s_rad = s_ang / np.pi

    bandwidth = s_ang / 2

    # s_rad = 2 * bandwidth / np.pi

    print("s_ang = {0}".format(s_ang))
    print("s_rad = {0}".format(s_rad))
    print("bandwidth = {0}".format(bandwidth))

    # maximum expected offset of COM
    # displacement samples k
    # pho_max is rho_max in the paper
    k = rho_max / pixel_sampling
    b = rho_max / 2

    # small value to avoid duplications
    eps = np.pi / (2 * k)

    # bandwiths maximum abs values of m1,h1,mm

    s_ksi = np.pi * k
    s_eta = k

    bound_m1 = np.floor(s_ksi / 2.0)
    bound_h1 = np.floor(s_eta / 2.0)
    bound_mm = np.round(bandwidth)

    m1_net = np.arange(-bound_m1, bound_m1, dtype='int32')
    h1_net = np.arange(-bound_h1, bound_h1, dtype='int32')
    mm_net = np.arange(0, bound_mm, dtype='int32')

    # Grid of angles for formula 10, 11, grid of x parameter

    theta_net = np.linspace(0, 2 * np.pi, int(s_ang))
    u_net = np.linspace(0, image_radius, int(s_rad))
    x_net = np.linspace(0, bandwidth / image_radius, int(s_rad))

    # final parameters of motion
    ksi_net = np.linspace(-np.pi, np.pi, len(m1_net))
    eta_net = np.linspace(-np.pi, np.pi, len(h1_net))
    omega_net = np.linspace(-np.pi, np.pi, len(mm_net))
    return m1_net, h1_net, mm_net, theta_net, u_net, x_net, omega_net, ksi_net, eta_net, eps, b, bandwidth


def laguerre_functions_precompute(alphas: List[int],
                                  x: np.ndarray, lag_func_num: int = 40, lag_scale: float = 3):
    laguerre_functions = {}
    lag_object = Laguerre()
    for alpha in alphas:
        lag_functions = lag_object.create_functions_x2_2sqrtx(lag_func_num, alpha, x * lag_scale, 1)
        laguerre_functions[alpha] = lag_functions
    return laguerre_functions


def image_fbt_precompute(image: np.ndarray, alphas: List[int], theta_net: np.ndarray,
                         u_net: np.ndarray, x_net: np.ndarray, method: str = 'fbm',
                         lag_func_num: int = 40, lag_scale: float = 3, lag_num_dots: int = 2000,
                         additional_params: Optional[Dict[str, Any]] = None):
    assert method in ['fbm', 'fbm_laguerre', 'fast_fbm_laguerre'], 'Choose one of the set options for the method!'

    fbt_arr = {}
    if method in ['fbm_laguerre', 'fast_fbm_laguerre'] and 'laguerre_functions' in additional_params:
        laguerre_functions = additional_params['laguerre_functions']
    else:
        laguerre_functions = None

    for alpha in alphas:
        if method == 'fbm':
            Fm = FBT(image, alpha, x_net, u_net, theta_net)
        elif method == 'fbm_laguerre':
            Fm = FBT_Laguerre(image, abs(alpha), x_net, u_net, theta_net, lag_func_num,
                              scale=lag_scale, num_dots=lag_num_dots,
                              out_lag_functions=laguerre_functions[abs(alpha)])
            if alpha < 0:
                Fm *= (-1) ** abs(alpha)
        else:
            if 'lag_zeros' in additional_params:
                zeros = additional_params['lag_zeros'][abs(alpha)]
            else:
                zeros = None
            Fm = FBT_Laguerre_fast(image, abs(alpha),
                                   x_net, u_net, theta_net,
                                   lag_func_num, scale=lag_scale,
                                   num_dots=lag_num_dots, zeros=zeros,
                                   lag_functions=laguerre_functions[abs(alpha)])
            # if alpha == 0:
            #     plt.figure()
            #     plt.plot(np.arange(len(Fm)), Fm)
            #     plt.title('alpha 0 fbt fast')
            #     plt.figure()
            #     plt.plot(np.arange(len(laguerre_functions[0][0])), laguerre_functions[0][0])
            #     plt.title('Laguerre_func')
            if alpha < 0:
                Fm *= (-1) ** abs(alpha)

        fbt_arr[alpha] = Fm
    return fbt_arr


def laguerre_zeros_precompute(alphas: List[int], lag_func_num: int = 40, abort_after: float = 1.):
    lag_object = Laguerre()
    zeros_lag = {}
    for alpha in tqdm(alphas):
        if alpha in zeros_lag:
            continue
        zeros = lag_object.laguerre_zeros(lag_func_num, alpha, abort_after=abort_after)
        zeros_lag[alpha] = zeros
    return zeros_lag


def fbm_registration(im1: np.ndarray, im2: np.ndarray,
                     image_radius: int = 128, p_s: float = 2., pho_max: int = 14,
                     method: str = 'fbm', lag_func_num: int = 40, lag_scale: float = 3, lag_num_dots: int = 2000,
                     shift_by_mask: bool = False, masks: List[np.ndarray] = None,
                     additional_params: Optional[Dict[str, Any]] = None):
    if additional_params is None:
        additional_params = {}

    assert method in ['fbm', 'fbm_laguerre', 'fast_fbm_laguerre'], 'Choose one of the set options for the method!'

    if 'integration_intervals' in additional_params:
        m1_net, h1_net, mm_net, theta_net, u_net, x_net, omega_net, ksi_net, eta_net, eps, b, bandwidth \
            = additional_params['integration_intervals']
    else:
        m1_net, h1_net, mm_net, theta_net, u_net, x_net, omega_net, ksi_net, eta_net, eps, b, bandwidth = \
            set_integration_intervals(image_radius, p_s, pho_max)

    # maxrad = im1.shape[0] ** 2 + im1.shape[1] ** 2
    # maxrad **= 0.5
    # maxrad = np.ceil(maxrad).astype(int)
    maxrad = image_radius

    if 'polar_fixed' in additional_params:
        pol1 = additional_params['polar_fixed']
    else:
        pol1 = polar_trfm(im1, int(2 * bandwidth), int(2 * bandwidth / np.pi), maxrad)

    # plt.figure()
    # plt.imshow(pol1)

    if 'precomputed_fbt_fixed' in additional_params:
        Fm_arr = additional_params['precomputed_fbt_fixed']
    else:
        alphas = []
        for it_m1 in range(len(m1_net)):
            m1 = m1_net[it_m1]
            for it_h1 in range(len(h1_net)):
                h1 = h1_net[it_h1]
                for it_mm in range(len(mm_net)):
                    mm = mm_net[it_mm]
                    if m1 + h1 + mm in alphas:
                        continue
                    alphas.append(m1 + h1 + mm)
        # print('Fm precompute')
        if method in ['fbm_laguerre', 'fast_fbm_laguerre'] and \
                'laguerre_functions' not in additional_params:
            additional_params['laguerre_functions'] = laguerre_functions_precompute(alphas, x_net,
                                                                                    lag_func_num, lag_scale)
        Fm_arr = image_fbt_precompute(pol1, alphas, theta_net, u_net, x_net,
                                      method, lag_func_num, lag_scale,
                                      lag_num_dots,
                                      additional_params)

    if 'precomputed_c1_coefs' in additional_params:
        c1_coefs = additional_params['precomputed_c1_coefs']
    else:
        c1_coefs = np.zeros((len(m1_net), len(x_net)))
        for it_m1 in range(len(m1_net)):
            m1 = m1_net[it_m1]
            c1 = sp.special.jv(m1, b * x_net) * x_net
            c1_coefs[it_m1, :] = c1
            # plt.figure
            # plt.plot(x_net, c1, )
            # plt.title('m1={0}'.format(m1))
            # plt.show()
            # time.sleep(3)

    if 'precomputed_c2_coefs' in additional_params:
        c2_coefs = additional_params['precomputed_c2_coefs']
    else:
        c2_coefs = np.zeros((len(h1_net), len(x_net)))
        for it_m1 in range(len(m1_net)):
            for it_h1 in range(len(h1_net)):
                h1 = h1_net[it_h1]
                if it_m1 == 0:
                    c2 = sp.special.jv(h1, b * x_net)
                    c2_coefs[it_h1, :] = c2

    if shift_by_mask:
        assert masks is not None, 'unable to perform shift by mask'
        mask1, mask2 = masks

        props = regionprops(label(mask1 > 0))
        c1y, c1x = props[0].centroid
        props = regionprops(label(mask2 > 0))
        c2y, c2x = props[0].centroid

        mat_shift_vec = get_translation_mat(c1x - c2x, c1y - c2y)
        im2 = cv2.warpAffine(
            im2, get_mat_2x3(mat_shift_vec),
            (im2.shape[1], im2.shape[0])
        )

    Tf = np.zeros((len(m1_net), len(h1_net), len(mm_net)), dtype='complex')

    if 'polar_moving' in additional_params:
        pol2 = additional_params['polar_moving']
    else:
        pol2 = polar_trfm(im2, int(2 * bandwidth), int(2 * bandwidth / np.pi), maxrad)

    # plt.figure()
    # plt.imshow(pol2)

    if 'precomputed_fbt_moving' in additional_params:
        Gm_arr = additional_params['precomputed_fbt_moving']
    else:
        # print('Gm precompute')
        Gm_arr = image_fbt_precompute(pol2, mm_net, theta_net, u_net, x_net, method,
                                      lag_func_num, lag_scale, lag_num_dots,
                                      additional_params)

    for it_m1 in range(len(m1_net)):
        c1 = c1_coefs[it_m1, :]
        m1 = m1_net[it_m1]
        for it_h1 in range(len(h1_net)):
            h1 = h1_net[it_h1]
            c2 = c2_coefs[it_h1, :] * c1
            for it_mm in range(len(mm_net)):
                mm = mm_net[it_mm]
                coef = 2 * np.pi * np.exp(1j * (h1 + mm) * eps)
                Fm = Fm_arr[m1 + h1 + mm]
                Gm = Gm_arr[mm]
                func = Fm * np.conj(Gm) * c2
                Tf[it_m1, it_h1, it_mm] = np.trapz(func, x_net) - 1j * np.trapz(np.imag(func), x_net)
                Tf[it_m1, it_h1, it_mm] *= coef

    T = np.fft.ifftn(Tf)
    [iksi, ieta, iomega] = np.unravel_index(np.argmax(T), Tf.shape)

    ksi = ksi_net[iksi]
    eta = eta_net[ieta]
    omega = omega_net[iomega]

    result = {'ksi': ksi, 'eta': eta, 'omega': omega, 'pho_max': pho_max / 2, 'eps': eps}
    if shift_by_mask:
        result['center_shift'] = mat_shift_vec

    return result


def apply_transform2(image, transform_dict):
    ksi = transform_dict['ksi']
    eta_prime = transform_dict['eta']
    omega_prime = transform_dict['omega']
    eps = transform_dict['eps']
    pho_max = transform_dict['pho_max']

    eta = eta_prime - ksi
    omega = omega_prime - eta_prime

    h, w = image.shape[:2]
    center = [w // 2, h // 2]

    mat_trans_center = get_translation_mat(*center)
    mat_rot_ksi = get_rotation_mat(np.pi - ksi, radians=True)
    mat_rot_eta = get_rotation_mat(np.pi - eta, radians=True)
    mat_rot_omaga = get_rotation_mat(np.pi - omega, radians=True)
    mat_trans_b = get_translation_mat(-pho_max / 2, 0)

    mat_all = mat_trans_center @ mat_rot_ksi @ mat_trans_b @ mat_rot_eta @ mat_trans_b @ mat_rot_omaga @ mat_inv(mat_trans_center)

    final_image = cv2.warpAffine(
        image, get_mat_2x3(mat_all),
        (image.shape[1], image.shape[0]))


    return final_image


def apply_transform1(image, transform_dict):
    ksi = transform_dict['ksi']
    eta_prime = transform_dict['eta']
    omega_prime = transform_dict['omega']
    eps = transform_dict['eps']
    pho_max = transform_dict['pho_max']

    eta = eta_prime - ksi
    omega = omega_prime - eta_prime

    phi_1 = ksi
    phi_2 = eta + eps
    psi_1 = 0
    psi_2 = omega
    rho_1 = pho_max / 2.0
    rho_2 = pho_max / 2.0

    t_x = rho_1 * math.cos(phi_1) + rho_2 * math.cos(phi_1 + phi_2 + psi_1)
    t_y = rho_1 * math.sin(phi_1) + rho_2 * math.sin(phi_1 + phi_2 + psi_1)
    t = [t_x, t_y]

    # angle = np.pi-(phi_1 + phi_2 + psi_1 + psi_2)
    angle = (phi_1 + phi_2 + psi_1 + psi_2)

    h, w = image.shape[:2]
    center = [w // 2, h // 2]
    trans_mat_center = get_translation_mat(-center[0], -center[1])
    trans_mat = get_translation_mat(t_x, t_y)
    rot_mat = get_rotation_mat(angle, True)

    # 
    warp_mat_1 = mat_inv(trans_mat_center) @ rot_mat @ trans_mat_center
    im2_1 = warp_img(image, warp_mat_1)
    # show_img_ovl(image, im2_1)
    print('warp_mat_1')
    print(trans_mat_center @ warp_mat_1 @ mat_inv(trans_mat_center))

    #
    warp_mat_2 = mat_inv(trans_mat_center) @ trans_mat @ rot_mat @ trans_mat_center
    im2_2 = warp_img(image, warp_mat_2)
    # show_img_ovl(image, im2_2)
    print('warp_mat_2')
    print(trans_mat_center @ warp_mat_2 @ mat_inv(trans_mat_center))

    return im2_2, im2_1, t, angle


def apply_transform(image, transform_dict, center=None, verbose=False):
    h, w = image.shape[:2]
    if center is None:
        center = [w // 2, h // 2]
    ksi = transform_dict['ksi']
    eta_prime = transform_dict['eta']
    omega_prime = transform_dict['omega']
    eps = transform_dict['eps']
    pho_max = transform_dict['pho_max']

    eta = eta_prime - ksi
    omega = omega_prime - eta_prime

    if 'center_shift' in transform_dict:
        image_shifted = cv2.warpAffine(
            image, get_mat_2x3(transform_dict['center_shift']),
            (w, h)
        )
    else:
        image_shifted = image

    mat_trans_center = get_translation_mat(*center)
    mat_rot_ksi = get_rotation_mat(normalize_alpha(ksi), radians=True)
    mat_trans_b = get_translation_mat(pho_max / 2, 0)

    mat_o_p = mat_trans_b @ mat_trans_center @ mat_rot_ksi @ mat_inv(mat_trans_b)

    P_coords = get_mat_2x3(mat_o_p) @ np.array([0, 0, 1])

    if verbose:
        print('P_coords', P_coords)

    mat_trans_p = get_translation_mat(P_coords[0], P_coords[1])
    mat_rot_eta = get_rotation_mat(normalize_alpha(eta), True)
    mat_trans_b = get_translation_mat(pho_max, 0)

    mat_trans_r = mat_inv(mat_trans_b) @ mat_trans_p @ mat_rot_eta @ mat_trans_b

    O_coords = get_mat_2x3(mat_trans_r) @ np.array([0, 0, 1])
    shift_vec = [O_coords[0] - center[0], O_coords[1] - center[1]]

    mat_shift_vec = get_translation_mat(*shift_vec)
    im_shifted_o = cv2.warpAffine(
        image_shifted, get_mat_2x3(mat_shift_vec),
        (image_shifted.shape[1], image_shifted.shape[0])
    )
    if verbose:
        plt.figure()
        plt.imshow((np.stack([image, im_shifted_o, image * 0], -1) * 255).astype('uint8'))
        plt.scatter(*P_coords)
        plt.scatter(*center)
        plt.scatter(*O_coords)

    mat_trans_o = get_translation_mat(*O_coords)
    mat_rot_omega = get_rotation_mat(normalize_alpha(omega + eps),
                                      radians=True)

    mat_trans_rot = mat_trans_o @ mat_rot_omega @ mat_inv(mat_trans_o)

    final_image = cv2.warpAffine(
        im_shifted_o, get_mat_2x3(mat_trans_rot),
        (im_shifted_o.shape[1], im_shifted_o.shape[0])
    )
    if verbose:
        plt.figure()
        plt.imshow((np.stack([image, final_image, image * 0], -1) * 255).astype('uint8'))
    return final_image
