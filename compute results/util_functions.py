import numpy as np
import mrcfile
from matrix_utils import *
import cv2


def normalize(im):
    return (im - im.min()) / (im.max() - im.min())


def shift(im, vec):
    mat_trans = get_translation_mat(*vec)
    return cv2.warpAffine(im, get_mat_2x3(mat_trans),
                          (im.shape[1], im.shape[0]))


def rotate(im, angle, center=None):
    h, w = im.shape[:2]
    if center is None:
        center = np.array([w // 2, h // 2])
    mat_trans_minus_center = get_translation_mat(-center[0], -center[1])
    mat_rot = get_rotation_mat(angle, radians=False)
    mat_trans_center = get_translation_mat(*center)
    return cv2.warpAffine(im, get_mat_2x3(mat_trans_center @ mat_rot @ mat_trans_minus_center),
                          (im.shape[1], im.shape[0]))


def apply_transform(image, transform_dict, rotate_only=False):
    h, w = image.shape[:2]
    ksi = transform_dict['ksi']
    etta_prime = transform_dict['etta']
    omegga_prime = transform_dict['omegga']
    etta = etta_prime - ksi
    omegga = omegga_prime - etta_prime
    eps = transform_dict['eps']
    com_offset = transform_dict['com_offset']

    rho = (2 * com_offset ** 2 * (1 + np.cos(etta + eps))) ** 0.5
    tx, ty = rho * np.cos(ksi), rho * np.sin(ksi)
    alpha = etta + eps + omegga + ksi
    #     print('alpha', np.degrees(alpha))

    im_reg = rotate(image, -np.degrees(alpha))
    if not rotate_only:
        im_reg = shift(im_reg, (-tx, -ty))

    return im_reg, [np.degrees(alpha), tx, ty]


def save_arr_mrc(fname, arr):
    with mrcfile.new(fname, overwrite=True) as mrc:
        mrc.set_data(np.array(arr))


def convert_angle_to_interval(value, interval=[-180, 180]):
    minv, maxv = interval
    interval_range = maxv-minv
    if value > maxv:
        value -= interval_range
    if value < minv:
        value += interval_range
    return value