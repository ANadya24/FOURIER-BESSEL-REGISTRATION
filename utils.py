import numpy as np
from scipy.ndimage import geometric_transform
from skimage.transform import AffineTransform, warp


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
    # theta, radius = np.meshgrid(theta_int, r_int)

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


def shift(image, vector):
    transform = AffineTransform(translation=vector)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)

    shifted = shifted.astype(image.dtype)
    return shifted


def iou(im1, im2):
    im1 = im1 > 0
    im2 = im2 > 0
    inter = (im1 * im2).sum()
    union = (im1 | im2).sum()
    return inter / (union + 1e-7)


def normalize_alpha(alpha, radians=True):
    if not radians:
        alpha = np.radians(alpha)
    return min((np.pi - alpha) % np.pi, np.pi - ((np.pi - alpha) % np.pi))