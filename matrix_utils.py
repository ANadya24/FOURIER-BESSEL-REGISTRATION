import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd

inference_size = 1024


def show_img_pair(source, target):
    fig, ax = plt.subplots(1,2, figsize=(30,10))
    ax[0].imshow(source, cmap='gray')
    ax[1].imshow(target, cmap='gray')
    plt.show()


def pad_img_bottom_right(img, new_shape):
    pad_source_x = max(0, new_shape[0] - img.shape[0])
    pad_source_y = max(0, new_shape[1] - img.shape[1])
    return np.pad(img, ((0, pad_source_x), (0, pad_source_y)), constant_values=0)


def overlay_img_pair(source, target):
    max_shape = (max(source.shape[0], target.shape[0]), max(source.shape[1], target.shape[1]))
    source = pad_img_bottom_right(source, max_shape)
    target = pad_img_bottom_right(target, max_shape)
    overlay_img = np.dstack((source, target, np.zeros(source.shape)))
    plt.imshow(overlay_img)
    plt.show()


def get_translation_mat(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]).astype(np.float32)


def get_rotation_mat(angle, radians=False):
    if radians:
        theta = angle
    else:
        theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).astype(np.float32)


def get_scale_mat(s):
    return np.array([[s,  0.,  0.],
                     [0.,  s,  0.],
                     [0.,  0.,  1.]]).astype(np.float32)


def mat_inv(mat):
    return np.linalg.inv(mat)


def get_mat_2x3(mat):
    return mat[:-1,:]


def resample_image(image, resample_ratio):
    y_size, x_size = image.shape
    new_y_size, new_x_size = int(y_size / resample_ratio), int(x_size / resample_ratio)
    grid_x, grid_y = np.meshgrid(np.arange(new_x_size), np.arange(new_y_size))
    grid_x = grid_x * (x_size / new_x_size)
    grid_y = grid_y * (y_size / new_y_size)
    resampled_image = nd.map_coordinates(image, [grid_y, grid_x], cval=0, order=3)
    return resampled_image


def resample_images(source, target, resample_factor):
    gaussian_sigma = resample_factor / 1.25
    smoothed_source = nd.gaussian_filter(source, gaussian_sigma)
    smoothed_target = nd.gaussian_filter(target, gaussian_sigma)
    resampled_source = resample_image(smoothed_source, resample_factor)
    resampled_target = resample_image(smoothed_target, resample_factor)
    return resampled_source, resampled_target


def pad_single(image, new_shape):
    y_size, x_size = image.shape
    y_pad = ((int(np.floor((new_shape[0] - y_size)/2))), int(np.ceil((new_shape[0] - y_size)/2)))
    x_pad = ((int(np.floor((new_shape[1] - x_size)/2))), int(np.ceil((new_shape[1] - x_size)/2)))
    new_image = np.pad(image, (y_pad, x_pad), constant_values=0)
    return new_image


def pad_images_np(source, target):
    y_size_source, x_size_source = source.shape
    y_size_target, x_size_target = target.shape
    new_y_size = max(y_size_source, y_size_target)
    new_x_size = max(x_size_source, x_size_target)
    new_shape = (new_y_size, new_x_size)

    padded_source = pad_single(source, new_shape)
    padded_target = pad_single(target, new_shape)
    padding_source = ((new_shape[0]-source.shape[0]) / 2, (new_shape[1]-source.shape[1]) / 2)
    padding_target = ((new_shape[0]-target.shape[0]) / 2, (new_shape[1]-target.shape[1]) / 2)
    return padded_source, padded_target, padding_source, padding_target