import numpy as np
import scipy as sp
from tqdm import tqdm
import pickle
import os
import pandas as pd
from skimage import registration as im_reg_met

from utils import shift, polar_trfm
from registration import (
    set_integration_intervals,
    laguerre_zeros_precompute,
    image_fbt_precompute,
    fbm_registration,
    laguerre_functions_precompute,
    mu_precompute
)

from util_functions import (normalize,
                            apply_transform,
                            mean_std_normalize
                            )


def run_fast_fbm_laguerre(seq, func_parameters, image_radius, pixel_sampling, com_offset_initial,
                          lag_func_num, lag_scale, lag_num_dots, normalization='standard',
                          do_2nd_step=False):
    assert normalization in ['standard', 'mean_std']
    if normalization == 'standard':
        normalize_func = normalize
    else:
        normalize_func = mean_std_normalize
    df = pd.DataFrame(columns=['x', 'y', 'ang', 'ksi', 'eta_prime', 'omega_prime', 'dft_x', 'dft_y'])
    fbm_seq = [seq[0].copy()]
    fbm_seq_shift = [seq[0].copy()]

    fixed_image = normalize_func(sp.ndimage.gaussian_filter(seq[0].copy(), 1.3))

    func_parameters = fixed_image_precompute(image=fixed_image,
                                             additional_params=func_parameters,
                                             method='fast_fbm_laguerre',
                                             image_radius=image_radius,
                                             lag_func_num=lag_func_num,
                                             lag_scale=lag_scale,
                                             lag_num_dots=lag_num_dots)

    for i in tqdm(range(1, len(seq))):
        reg1 = fbm_registration(fixed_image,
                                normalize_func(sp.ndimage.gaussian_filter(seq[i].copy(), 1.3)),
                                image_radius=image_radius, p_s=pixel_sampling, com_offset=com_offset_initial,
                                method='fast_fbm_laguerre', lag_func_num=lag_func_num, lag_scale=lag_scale,
                                lag_num_dots=lag_num_dots,
                                masks=None, shift_by_mask=False, additional_params=func_parameters)

        im_reg1, params = apply_transform(seq[i].copy(), reg1)
        fbm_seq.append(im_reg1.copy())

        if do_2nd_step:
            shifts = im_reg_met.phase_cross_correlation(fixed_image,
                                                        normalize_func(sp.ndimage.gaussian_filter(im_reg1.copy(), 1.3)),
                                                        upsample_factor=100)[0][::-1]

            fbm_seq_shift.append(shift(im_reg1, -shifts))
        else:
            shifts = [0., 0.]

        founded_values = [params[1], params[2], params[0], reg1['ksi'],
                          reg1['etta'], reg1['omegga'], shifts[0], shifts[1]]
        df.loc[i - 1] = founded_values

    fbm_seq = np.array(fbm_seq)
    return df, fbm_seq, fbm_seq_shift


def run_fbm(seq, func_parameters, image_radius, pixel_sampling, com_offset_initial, normalization='standard',
            do_2nd_step=False):
    assert normalization in ['standard', 'mean_std']
    if normalization == 'standard':
        normalize_func = normalize
    else:
        normalize_func = mean_std_normalize
    df = pd.DataFrame(columns=['x', 'y', 'ang', 'ksi', 'eta_prime', 'omega_prime', 'dft_x', 'dft_y'])
    fbm_seq = [seq[0].copy()]
    fbm_seq_shift = [seq[0].copy()]
    fixed_image = normalize_func(sp.ndimage.gaussian_filter(seq[0].copy(), 1.3))
    func_parameters = fixed_image_precompute(image=fixed_image,
                                             additional_params=func_parameters,
                                             method='fbm',
                                             image_radius=image_radius)
    for i in tqdm(range(1, len(seq))):
        reg1 = fbm_registration(fixed_image,
                                normalize_func(sp.ndimage.gaussian_filter(seq[i].copy(), 1.3)),
                                image_radius=image_radius, p_s=pixel_sampling, com_offset=com_offset_initial,
                                method='fbm', masks=None, shift_by_mask=False,
                                additional_params=func_parameters)
        im_reg1, params = apply_transform(seq[i].copy(), reg1)
        fbm_seq.append(im_reg1.copy())

        if do_2nd_step:
            shifts = im_reg_met.phase_cross_correlation(fixed_image,
                                                        normalize_func(sp.ndimage.gaussian_filter(im_reg1.copy(), 1.3)),
                                                        upsample_factor=100)[0][::-1]

            fbm_seq_shift.append(shift(im_reg1, -shifts))
        else:
            shifts = [0., 0.]
        founded_values = [params[1], params[2], params[0], reg1['ksi'],
                          reg1['etta'], reg1['omegga'], shifts[0], shifts[1]]
        df.loc[i - 1] = founded_values

    fbm_seq = np.array(fbm_seq)
    return df, fbm_seq, fbm_seq_shift


def run_fbm_laguerre(seq, func_parameters, image_radius, pixel_sampling,
                     com_offset_initial, lag_func_num, lag_scale, lag_num_dots, normalization='standard',
                     do_2nd_step=False, fixed_mean=False):
    assert normalization in ['standard', 'mean_std']
    if normalization == 'standard':
        normalize_func = normalize
    else:
        normalize_func = mean_std_normalize
    df = pd.DataFrame(columns=['x', 'y', 'ang', 'ksi', 'eta_prime', 'omega_prime', 'dft_x', 'dft_y'])
    fbm_seq = [seq[0].copy()]
    fbm_seq_shift = [seq[0].copy()]
    fixed_image = normalize_func(sp.ndimage.gaussian_filter(seq[0].copy(), 1.3))
    if not fixed_mean:
        func_parameters = fixed_image_precompute(image=fixed_image,
                                                 additional_params=func_parameters,
                                                 method='fbm_laguerre',
                                                 image_radius=image_radius,
                                                 lag_func_num=lag_func_num,
                                                 lag_scale=lag_scale,
                                                 lag_num_dots=lag_num_dots)
    for i in tqdm(range(1, len(seq))):
        reg1 = fbm_registration(fixed_image,
                                normalize_func(sp.ndimage.gaussian_filter(seq[i].copy(), 1.3)),
                                image_radius=image_radius, p_s=pixel_sampling, com_offset=com_offset_initial,
                                method='fbm_laguerre', masks=None, lag_num_dots=lag_num_dots, shift_by_mask=False,
                                lag_func_num=lag_func_num, lag_scale=lag_scale,
                                additional_params=func_parameters)
        im_reg1, params = apply_transform(seq[i].copy(), reg1)
        fbm_seq.append(im_reg1.copy())
        if fixed_mean:
            prev_fixed_image = fixed_image.copy()
            fixed_image = np.stack([fixed_image, normalize_func(im_reg1)], 0).mean(0)
            if prev_fixed_image.std() < fixed_image.std():
                fixed_image = prev_fixed_image

        if do_2nd_step:
            shifts = im_reg_met.phase_cross_correlation(fixed_image,
                                                        normalize_func(
                                                            sp.ndimage.gaussian_filter(im_reg1.copy(), 1.3)),
                                                        upsample_factor=100)[0][::-1]

            fbm_seq_shift.append(shift(im_reg1, -shifts))
        else:
            shifts = [0., 0.]
        founded_values = [params[1], params[2], params[0], reg1['ksi'],
                          reg1['etta'], reg1['omegga'], shifts[0], shifts[1]]
        df.loc[i - 1] = founded_values

    fbm_seq = np.array(fbm_seq)
    return df, fbm_seq, fbm_seq_shift


def fixed_image_precompute(image, additional_params, method, image_radius, lag_func_num=None, lag_scale=None,
                           lag_num_dots=None):
    if additional_params is None:
        additional_params = {}

    Im1, Ih1, Imm, theta_net, \
    u_net, x_net, omega_net, psi_net, eta_net, eps, b, bandwidth = additional_params['integration_intervals']

    if 'polar_fixed' in additional_params:
        pol1 = additional_params['polar_fixed']
    else:
        maxrad = image_radius
        pol1 = polar_trfm(image, int(2 * bandwidth), int(2 * bandwidth / np.pi), maxrad)

    if 'precomputed_fbt_fixed' not in additional_params:
        alphas = additional_params['alphas']

        if method in ['fbm_laguerre', 'fast_fbm_laguerre'] and \
                'laguerre_functions' not in additional_params:
            additional_params['laguerre_functions'] = laguerre_functions_precompute(alphas, x_net,
                                                                                    lag_func_num, lag_scale)
        if method in ['fbm']:
            Fm_arr = image_fbt_precompute(pol1, alphas, theta_net, u_net, x_net,
                                          method, additional_params=additional_params)
        else:
            Fm_arr = image_fbt_precompute(pol1, alphas, theta_net, u_net, x_net,
                                          method, lag_func_num, lag_scale,
                                          lag_num_dots,
                                          additional_params)
        additional_params['precomputed_fbt_fixed'] = Fm_arr
    return additional_params


def precompute_w_params(image_radius, pixel_sampling, com_offset_initial, lag_func_num,
                        lag_scale, lag_num_dots=1000, compute_zeros=False, verbose=True, n_jobs=4,
                        num_zeros=None):
    params = {}

    Im1, Ih1, Imm, theta_net, u_net, x_net, omega_net, psi_net, eta_net, eps, b, bandwidth = \
        set_integration_intervals(image_radius, pixel_sampling, com_offset_initial, verbose=verbose)
    params['integration_intervals'] = \
        Im1, Ih1, Imm, theta_net, u_net, x_net, omega_net, psi_net, eta_net, eps, b, bandwidth

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
    params['alphas'] = alphas

    if compute_zeros:
        if num_zeros is None:
            num_zeros = lag_func_num * 2

        if os.path.exists(f'cryo_laguerre_zeros_{len(alphas)}_{num_zeros}.pkl'):
            with open(f'cryo_laguerre_zeros_{len(alphas)}_{num_zeros}.pkl', 'rb') as file:
                params['lag_zeros'] = pickle.load(file)
        else:
            print('lag_func_num', lag_func_num)
            params['lag_zeros'] = laguerre_zeros_precompute(alphas, num_zeros, n_jobs=n_jobs)
            with open(f'cryo_laguerre_zeros_{len(alphas)}_{num_zeros}.pkl', 'wb') as file:
                pickle.dump(params['lag_zeros'], file)
        params['mus'] = mu_precompute(params['lag_zeros'], lag_func_num)

    laguerre_functions = laguerre_functions_precompute(alphas, x_net,
                                                       lag_func_num, lag_num_dots, lag_scale)
    params['laguerre_functions'] = laguerre_functions

    c1_coefs = np.zeros((len(Im1), len(x_net)))
    for it_m1 in range(len(Im1)):
        m1 = Im1[it_m1]
        c1 = sp.special.jv(m1, b * x_net) * x_net
        c1_coefs[it_m1, :] = c1
    params['precomputed_c1_coefs'] = c1_coefs

    c2_coefs = np.zeros((len(Ih1), len(x_net)))
    for it_m1 in range(len(Im1)):
        for it_h1 in range(len(Ih1)):
            h1 = Ih1[it_h1]
            if it_m1 == 0:
                c2 = sp.special.jv(h1, b * x_net)
                c2_coefs[it_h1, :] = c2
    params['precomputed_c2_coefs'] = c2_coefs

    ih1, imm = np.meshgrid(Imm, Ih1)
    coef = 2 * np.pi * np.exp(1j * (ih1 + imm) * eps)
    params['precomputed_coef_exp'] = coef

    return params
