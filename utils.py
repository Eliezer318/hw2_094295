from PIL import Image
import numpy as np

shape = (64, 64)


def numpy2img(image_array):
    return Image.fromarray(np.uint8(image_array)).resize(shape)


def centralize_image(img):
    image_array = np.array(img)
    dh, dw = image_array.shape
    image_array = image_array.T[((255 - image_array).sum(0) != 0)].T
    image_array = image_array[((255 - image_array).sum(1) != 0)]
    dh, dw = dh - image_array.shape[0], dw - image_array.shape[1]
    return numpy2img(image_array), dh, dw


def decentralize_image(img, dh, dw):
    image_array = np.array(img)
    new_image_array = np.zeros((image_array.shape[0] + dh, image_array.shape[1] + dw)) + 255
    new_image_array[dh // 2: image_array.shape[0] + dh // 2, dw // 2: image_array.shape[1] + dw // 2] = image_array
    return numpy2img(new_image_array)


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    imga = np.array(imga)
    imgb = np.array(imgb)
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros((max_height, total_width))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return numpy2img(new_img)


def combine_images(img1, img2):
    img1, img2 = img1.resize(shape), img2.resize(shape)
    n1_img, dh1, dw1 = centralize_image(img1)
    n2_img, dh2, dw2 = centralize_image(img2)
    combined_image = concat_images(n1_img, n2_img)
    dh, dw = max(dh1, dh2), max(dw1, dw2)
    combined_image = decentralize_image(combined_image, dh, dw)
    return combined_image


def corners_and_center(image):
    img = np.array(image)
    img = img.T[((255 - img).sum(0) != 0)].T
    img = img[((255 - img).sum(1) != 0)]
    h, w = img.shape
    tl, tr, bl, br, cen = (255 + np.zeros((5, 2 * h, 2 * w)))[:]
    tl[:h, :w] = img
    tr[:h, w:2 * w] = img
    bl[h:2 * h, :w] = img
    br[h:2 * h, w:2 * w] = img
    cen[h // 2:h + h // 2, w // 2: w + w // 2] = img
    tl, tr, bl, br, cen = [Image.fromarray(np.uint8(new_image_array)) for new_image_array in [tl, tr, bl, br, cen]]
    return tl, tr, bl, br, cen
