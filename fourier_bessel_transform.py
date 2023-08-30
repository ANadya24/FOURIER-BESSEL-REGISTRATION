import numpy as np
import scipy as sp
from laguerre import Laguerre
from function_scale import FunctionScale


def FBT(pol, m, x_net, u_net, theta_net):
    """
        compute formulas 10 and 11 from article
        Parameters:
        pol : image resampled to polar coordinates
        m : order of Bessel function (non integer give errors)
        x_net : vector of x from formula 10
        u_net, theta_net: polar grid of the given image
        return: vector of len(x_net)
    """

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


def normed_Hankel_trf(f, alpha, x_net, u_net):
    # интегрируем по u

    bessel = sp.special.jv(alpha, u_net.reshape(-1, 1).
                           dot(x_net.reshape(1, -1)))

    ff = bessel * f.reshape(-1, 1) * np.sqrt(u_net.reshape(-1, 1).dot(x_net.reshape(1, -1)))

    H = np.trapz(np.real(ff), u_net, axis=0) + 1j * np.trapz(np.imag(ff), u_net, axis=0)
    return H


def Hankel_trf(f, alpha, x_dest, x_src):

    Fm = ( 1 /np.sqrt(x_dest +1e-9)) * normed_Hankel_trf(np.array(f ) *np.sqrt(x_src), alpha, x_dest, x_src)
    return Fm


def Fourier_alpa_trf(pol, m, theta_net):
    f1 = np.exp(-1j * m * theta_net)
    f2 = pol * f1.reshape(1, -1)
    #     f2 = np.fft.fft(pol*np.exp(-1j*m), axis=1)
    fm = np.trapz(np.real(f2), theta_net, axis=1) + 1j * np.trapz(np.imag(f2), theta_net, axis=1)
    fm = fm / (2 * np.pi)
    return fm


def FBT_Hankel(pol, m, x_dest, x_src, theta_net):
    """
        compute formulas 10 and 11 from article
        Parameters:
        pol : image resampled to polar coordinates
        m : order of Bessel function (non integer give errors)
        x_net : vector of x from formula 10
        u_net, theta_net: polar grid of the given image
        return: vector of len(x_net)
    """

    fm = Fourier_alpa_trf(pol, m, theta_net)
    Fm = Hankel_trf(fm, m, x_dest, x_src)

    return Fm


def FBT_Laguerre(pol, m, x_dest, x_src, theta_net, func_num, scale=1, num_dots=2000, out_lag_functions=None):
    """
        compute formulas 10 and 11 from article
        Parameters:
        pol : image resampled to polar coordinates
        m : order of Bessel function (non integer give errors)
        x_net : vector of x from formula 10
        u_net, theta_net: polar grid of the given image
        return: vector of len(x_net)
    """

    fm = Fourier_alpa_trf(pol, m, theta_net)

    lag_object = Laguerre()
    func_scale = FunctionScale()

    if num_dots == -1:
        x_src_multi_dot = x_src
    else:
        x_src_multi_dot = np.linspace(x_src.min(), x_src.max(), num_dots)

    fm_interp = func_scale.interpolate_2_new_grid(x_src, fm, x_src_multi_dot)
    fm_hat = func_scale.func_sqrtx(x_src_multi_dot, fm_interp)
    gm_hat = func_scale.x_compress_function(x_src_multi_dot, fm_hat, scale)
    lag_functions = lag_object.create_functions_x2_2sqrtx(func_num, m, x_src_multi_dot, 1)
    hankel_coefs = lag_object.transform_forward(func_num, gm_hat, lag_functions, x_src_multi_dot, 1)

    max_value = x_dest.max()
    if x_dest.max() < 10:
        max_value *= 5
    x_dest_multi_dot = np.linspace(x_dest.min(), max_value, num_dots)
    if out_lag_functions is None:
        lag_functions = lag_object.create_functions_x2_2sqrtx(func_num, m, x_dest_multi_dot * scale, 1)
    else:
        lag_functions = out_lag_functions
    Fm_forward = np.array(lag_object.transform_backward(func_num, hankel_coefs, lag_functions))
    mask = np.abs(x_dest_multi_dot) > 1e-4
    Fm_forward[mask] = Fm_forward[mask] / x_dest_multi_dot[mask] ** 0.5

    Fm = Fm_forward * scale
    Fm = func_scale.interpolate_2_new_grid(x_dest_multi_dot, Fm, x_dest)

    return Fm

def FBT_Laguerre_fast(pol, m, x_dest, x_src, theta_net, func_num, scale=1, num_dots=2000,
                      zeros=None, return_zeros=False, lag_functions=None, num_zeros=None):
    """
        compute formulas 10 and 11 from article
        Parameters:
        pol : image resampled to polar coordinates
        m : order of Bessel function (non integer give errors)
        x_net : vector of x from formula 10
        u_net, theta_net: polar grid of the given image
        return: vector of len(x_net)
    """

    fm = Fourier_alpa_trf(pol, m, theta_net)
    lag_object = Laguerre()
    func_scale = FunctionScale()

    if zeros is None:
        if num_zeros is None:
            num_zeros = func_num + 1
        zeros = lag_object.laguerre_zeros(num_zeros, m)

    if num_dots == -1:
        x_src_multi_dot = x_src
    else:
        x_src_multi_dot = np.linspace(x_src.min(), x_src.max(), num_dots)
    fm_interp = func_scale.interpolate_2_new_grid(x_src, fm, x_src_multi_dot)
    fm_hat = func_scale.func_sqrtx(x_src_multi_dot.copy(), fm_interp)
    gm_hat = func_scale.x_compress_function(x_src_multi_dot.copy(), fm_hat, scale)

    hankel_fast_coefs = lag_object.transform_forward_fast_x2sqrtx_laguerre_quad(zeros, func_num, m,
                                                                                gm_hat, x_src_multi_dot)
    max_value = x_dest.max()
    if x_dest.max() < 10:
        max_value *= 5
    x_dest_multi_dot = np.linspace(x_dest.min(), max_value, num_dots)
    if lag_functions is None:
        lag_functions = lag_object.create_functions_x2_2sqrtx(func_num, m, x_dest_multi_dot * scale, 1)
    Fm = np.array(lag_object.transform_backward(func_num, hankel_fast_coefs, lag_functions))

    mask = np.abs(x_dest_multi_dot) > 1e-4
    Fm[mask] = Fm[mask] / x_dest_multi_dot[mask]**0.5

    Fm *= scale
    Fm = func_scale.interpolate_2_new_grid(x_dest_multi_dot, Fm, x_dest)

    if return_zeros:
        return Fm, zeros
    return Fm