from typing import List, Optional, Dict, Any
import numpy as np
from skimage.measure import label, regionprops
import cv2
import scipy as sp
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import polar_trfm, normalize_alpha
from matrix_utils import (
    get_translation_mat,
    get_mat_2x3,
    get_rotation_mat, mat_inv
)
from fourier_bessel_transform import FBT, FBT_Laguerre, FBT_Laguerre_fast
from laguerre import Laguerre


def set_integration_intervals(image_radius: int = 128,
                              pixel_sampling: float = 1., com_offset: float = 7.,
                              verbose: bool = False):
    '''
    Init the image_radius A, displacement samples k
    :param image_radius: radius of the image
    :param pixel_sampling:
    :param com_offset: maximum expected offset of COM (0.025 * image_diameter)
    :return:
    '''

    # image_radius = 2 * bandwidth * pixel_sampling / np.pi
    # image_radius = 128

    if verbose:
        print("============")
        print(image_radius)

    # section 2.2.1: number of angular samples
    # to be used in formula 10, 11
    s_ang = np.pi * image_radius / pixel_sampling

    # section 2.2.1: number of radial samples
    # to be used in formula 10, 11
    s_rad = s_ang / np.pi

    bandwidth = s_ang / 2

    # s_rad = 2 * bandwidth / np.pi

    if verbose:
        print("s_ang = {0}".format(s_ang))
        print("s_rad = {0}".format(s_rad))
        print("bandwidth = {0}".format(bandwidth))

    # maximum expected offset of COM
    # displacement samples k
    # com_offset is rho_max in the paper
    k = com_offset / pixel_sampling
    b = com_offset / 2

    # small value to avoid duplications
    eps = np.pi / (2 * k)

    # bandwiths maximum abs values of m1,h1,mm

    bound_m1 = np.floor(2 * b * bandwidth / image_radius)
    bound_h1 = np.floor(2 * b * bandwidth / (np.pi * image_radius))
    bound_mm = np.round(bandwidth)

    Im1 = np.arange(-bound_m1, bound_m1+1, dtype='int32')
    Ih1 = np.arange(-bound_h1, bound_h1+1, dtype='int32')
    Imm = np.arange(0, bound_mm+1, dtype='int32')
    if verbose:
        print('len(Im1)', len(Im1), 'len(Ih1)', len(Ih1), 'len(Imm)', len(Imm))

    # Grid of angles for formula 10, 11, grid of x parameter

    theta_net = np.linspace(0, 2 * np.pi, int(s_ang))
    u_net = np.linspace(0, image_radius, int(s_rad))
    x_net = np.linspace(0, bandwidth / image_radius, int(s_rad))

    # final parameters of motion

    omega_net = np.linspace(0, 2 * np.pi, len(Imm))
    # psi_net = np.linspace(-np.pi, np.pi, int(4 * b * bandwidth / image_radius))
    # eta_net = np.linspace(-np.pi, np.pi, int(4 * b * bandwidth / (np.pi * image_radius)))
    ksi_net = np.linspace(0, 2 * np.pi, len(Im1)) #int(np.pi * k))
    eta_net = np.linspace(0, 2 * np.pi,len(Ih1)) #int(k))
    return Im1, Ih1, Imm, theta_net, u_net, x_net, omega_net, ksi_net, eta_net, eps, b, bandwidth


def laguerre_functions_precompute(alphas: List[int],
                                  x: np.ndarray, lag_func_num: int = 40,
                                  lag_num_dots: int = 1000, lag_scale: float = 3):
    laguerre_functions = {}
    lag_object = Laguerre()

    # max_value = x.max()
    # if x.max() < 10:
    #     max_value *= 5
    # x_grid = np.linspace(x.min(), max_value, lag_num_dots)
    for alpha in alphas:
        # if alpha >= 0 and alpha < 178:
        # if np.isinf(sp.special.gamma(abs(alpha))):
        #     continue
        if abs(alpha) not in laguerre_functions:
            lag_functions = lag_object.create_functions_x2_2sqrtx(lag_func_num, alpha, x * lag_scale, 1)
            laguerre_functions[alpha] = lag_functions
    return laguerre_functions


def laguerre_zeros_precompute(alphas: List[int], num_zeros: int = 40, abort_after: int = 5000, n_jobs:int = 4):
    lag_object = Laguerre()
    zeros_lag = {}

    def calc_zeros(alpha):
        if np.isinf(sp.special.gamma(abs(alpha))):
            return alpha, np.zeros(num_zeros)
        return alpha, lag_object.laguerre_zeros(num_zeros, alpha, abort_after=abort_after)

    print('Total alphas:', len(alphas))
    abs_alphas = np.unique(np.abs(alphas))
    print('Total abs alphas:', len(abs_alphas))

    results = Parallel(n_jobs=n_jobs)(delayed(calc_zeros)(alpha) for alpha in tqdm(abs_alphas))
    for alpha, zeros in results:
        zeros_lag[alpha] = zeros

    return zeros_lag


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
        skip = False
        if method == 'fbm':
            Fm = FBT(image, alpha, x_net, u_net, theta_net)
        else:
            # if np.isinf(sp.special.gamma(abs(alpha))):
            #     Fm = np.zeros(len(x_net))
            # else:
            if laguerre_functions is None:
                laguerre_func = None
            elif abs(alpha) in laguerre_functions:
                laguerre_func = laguerre_functions[abs(alpha)]
            else:
                skip = True
            if method == 'fbm_laguerre' and not skip:
                Fm = FBT_Laguerre(image, abs(alpha), x_net, u_net, theta_net, lag_func_num,
                                  scale=lag_scale, num_dots=lag_num_dots,
                                  out_lag_functions=laguerre_func)
                if alpha < 0:
                    Fm *= (-1) ** abs(alpha)
            elif not skip:
                if 'lag_zeros' in additional_params:
                    zeros = additional_params['lag_zeros'][abs(alpha)]
                else:
                    zeros = None
                Fm = FBT_Laguerre_fast(image, abs(alpha),
                                       x_net, u_net, theta_net,
                                       lag_func_num, scale=lag_scale,
                                       num_dots=lag_num_dots, zeros=zeros,
                                       lag_functions=laguerre_func)

                if alpha < 0:
                    Fm *= (-1) ** abs(alpha)
            else:
                Fm = np.zeros(len(x_net))

        fbt_arr[alpha] = Fm
    return fbt_arr


def fbm_registration(im1: np.ndarray, im2: np.ndarray,
                     image_radius: int = 128, p_s: float = 2., com_offset: int = 14,
                     method: str = 'fbm', lag_func_num: int = 40, lag_scale: float = 3, lag_num_dots: int = 2000,
                     shift_by_mask: bool = False, masks: List[np.ndarray] = None,
                     additional_params: Optional[Dict[str, Any]] = None, verbose: bool = False):
    if additional_params is None:
        additional_params = {}

    assert method in ['fbm', 'fbm_laguerre', 'fast_fbm_laguerre'], 'Choose one of the set options for the method!'

    if 'integration_intervals' in additional_params:
        Im1, Ih1, Imm, theta_net, u_net, x_net, omega_net, ksi_net, eta_net, eps, b, bandwidth \
            = additional_params['integration_intervals']
    else:
        Im1, Ih1, Imm, theta_net, u_net, x_net, omega_net, ksi_net, eta_net, eps, b, bandwidth = \
            set_integration_intervals(image_radius=image_radius,
                                      pixel_sampling=p_s, com_offset=com_offset, verbose=verbose)

    maxrad = image_radius

    if 'precomputed_fbt_fixed' in additional_params:
        Fm_arr = additional_params['precomputed_fbt_fixed']
    else:
        if 'alphas' in additional_params:
            alphas = additional_params['alphas']
        else:
            alphas = []
            for it_m1 in range(len(Im1)):
                m1 = Im1[it_m1]
                for it_h1 in range(len(Ih1)):
                    h1 = Ih1[it_h1]
                    for it_mm in range(len(Imm)):
                        mm = Imm[it_mm]
                        if m1 + h1 + mm in alphas:
                            continue
                        alphas.append(m1 + h1 + mm)

        pol1 = polar_trfm(im1, int(2 * bandwidth), int(2 * bandwidth / np.pi), maxrad)

        if method in ['fbm_laguerre', 'fast_fbm_laguerre'] and \
                'laguerre_functions' not in additional_params:
            additional_params['laguerre_functions'] = laguerre_functions_precompute(alphas=alphas,
                                                                                    x=x_net,
                                                                                    lag_func_num=lag_func_num,
                                                                                    lag_num_dots=lag_num_dots,
                                                                                    lag_scale=lag_scale)
        Fm_arr = image_fbt_precompute(image=pol1, alphas=alphas,
                                      theta_net=theta_net, u_net=u_net, x_net=x_net,
                                      method=method, lag_func_num=lag_func_num, lag_scale=lag_scale,
                                      lag_num_dots=lag_num_dots, additional_params=additional_params)

    if 'precomputed_c1_coefs' in additional_params:
        c1_coefs = additional_params['precomputed_c1_coefs']
    else:
        c1_coefs = np.zeros((len(Im1), len(x_net)))
        for it_m1 in range(len(Im1)):
            m1 = Im1[it_m1]
            c1 = sp.special.jv(m1, b * x_net) * x_net
            c1_coefs[it_m1, :] = c1

    if 'precomputed_c2_coefs' in additional_params:
        c2_coefs = additional_params['precomputed_c2_coefs']
    else:
        c2_coefs = np.zeros((len(Ih1), len(x_net)))
        for it_m1 in range(len(Im1)):
            for it_h1 in range(len(Ih1)):
                h1 = Ih1[it_h1]
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

    Tf = np.zeros((len(Im1), len(Ih1), len(Imm)), dtype='complex')

    # if 'polar_moving' in additional_params:
    #     pol2 = additional_params['polar_moving']
    # else:
    pol2 = polar_trfm(im2, int(2 * bandwidth), int(2 * bandwidth / np.pi), maxrad)

    # if 'precomputed_fbt_moving' in additional_params:
    #     Gm_arr = additional_params['precomputed_fbt_moving']
    # else:

    Gm_arr = image_fbt_precompute(image=pol2, alphas=Imm, theta_net=theta_net,
                                  u_net=u_net, x_net=x_net, method=method,
                                  lag_func_num=lag_func_num, lag_scale=lag_scale,
                                  lag_num_dots=lag_num_dots,
                                  additional_params=additional_params)

    if 'precomputed_coef_exp' in additional_params:
        coef = additional_params['precomputed_coef_exp']
    else:
        ih1, imm = np.meshgrid(Imm, Ih1)
        coef = 2 * np.pi * np.exp(1j * (ih1 + imm) * eps)

    for it_m1 in range(len(Im1)):
        c1 = c1_coefs[it_m1, :]
        m1 = Im1[it_m1]
        Tf[it_m1] = coef
        for it_h1 in range(len(Ih1)):
            h1 = Ih1[it_h1]
            c2 = c2_coefs[it_h1, :] * c1
            for it_mm in range(len(Imm)):
                mm = Imm[it_mm]
                Fm = Fm_arr[m1 + h1 + mm]
                Gm = Gm_arr[mm]
                func = Fm * np.conj(Gm) * c2
                Tf[it_m1, it_h1, it_mm] *= np.sum(func) * 0.5 * (x_net[1] - x_net[0])

    T = np.fft.ifftn(np.fft.ifftshift(Tf))
    [iksi, ietta, iomegga] = np.unravel_index(np.argmax(T), Tf.shape)

    ksi = ksi_net[iksi]
    etta = eta_net[ietta]
    omegga = omega_net[iomegga]

    result = {'ksi': ksi, 'etta': etta, 'omegga': omegga, 'com_offset': com_offset / 2, 'eps': eps}
    if shift_by_mask:
        result['center_shift'] = mat_shift_vec
    result['indices'] = iksi, ietta, iomegga
    return result


def apply_transform(image, transform_dict, center=None, verbose=False):
    h, w = image.shape[:2]
    if center is None:
        center = [w // 2, h // 2]
    ksi = transform_dict['ksi']
    etta = transform_dict['etta']
    omegga = transform_dict['omegga']
    eps = transform_dict['eps']
    com_offset = transform_dict['com_offset']

    if 'center_shift' in transform_dict:
        image_shifted = cv2.warpAffine(
            image, get_mat_2x3(transform_dict['center_shift']),
            (w, h)
        )
    else:
        image_shifted = image

    mat_trans_center = get_translation_mat(*center)
    mat_rot_ksi = get_rotation_mat(normalize_alpha(ksi), radians=True)
    mat_trans_b = get_translation_mat(com_offset, 0)

    mat_o_p = mat_trans_b @ mat_trans_center @ mat_rot_ksi @ mat_inv(mat_trans_b)

    P_coords = get_mat_2x3(mat_o_p) @ np.array([0, 0, 1])

    if verbose:
        print('P_coords', P_coords)

    mat_trans_p = get_translation_mat(P_coords[0], P_coords[1])
    mat_rot_etta = get_rotation_mat(normalize_alpha(etta), True)
    mat_trans_b = get_translation_mat(com_offset, 0)

    mat_trans_r = mat_inv(mat_trans_b) @ mat_trans_p @ mat_rot_etta @ mat_trans_b

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
    mat_rot_omegga = get_rotation_mat(normalize_alpha(omegga + eps),
                                      radians=True)

    mat_trans_rot = mat_trans_o @ mat_rot_omegga @ mat_inv(mat_trans_o)

    final_image = cv2.warpAffine(
        im_shifted_o, get_mat_2x3(mat_trans_rot),
        (im_shifted_o.shape[1], im_shifted_o.shape[0])
    )
    if verbose:
        plt.figure()
        plt.imshow((np.stack([image, final_image, image * 0], -1) * 255).astype('uint8'))
    return final_image
