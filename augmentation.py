from PIL import Image, ImageOps, ImageFilter
import numpy as np
import random
import os

from visualize import stats
from utils import combine_images, corners_and_center

shape = (64, 64)
N = 750


def mirror_and_flip(data_path='data', phase='train'):
    print('Augment data with mirroring and flipping')
    # mirror and get new label
    dict_count = stats(verbose=False)
    for source, target in [('iv', 'vi'), ('vi', 'iv')]:
        for idx, file_name in enumerate(os.listdir(f'{data_path}/{phase}/{source}')):
            if dict_count['train'][target] >= N:
                break
            dict_count['train'][target] += 1
            img = Image.open(f'{data_path}/{phase}/{source}/{file_name}')
            new_img = ImageOps.mirror(img)
            new_img.save(f'{data_path}/{phase}/{target}/{file_name.split(".")[0]}_mirror_{source}_{target}.png')

    # mirror
    dict_count = stats(verbose=False)
    for label in ['i', 'ii', 'iii', 'v', 'x']:
        for idx, file_name in enumerate(os.listdir(f'{data_path}/{phase}/{label}')[:N - dict_count['train'][label]]):
            if dict_count['train'][label] >= N:
                break
            dict_count['train'][label] += 1
            img = Image.open(f'{data_path}/{phase}/{label}/{file_name}')
            new_img = ImageOps.mirror(img)
            new_img.save(f'{data_path}/{phase}/{label}/{file_name.split(".")[0]}_mirror_regular.png')

    # flip
    dict_count = stats(verbose=False)
    for label in ['i', 'ii', 'iii', 'ix', 'x']:
        for idx, file_name in enumerate(os.listdir(f'{data_path}/{phase}/{label}')[:N - dict_count['train'][label]]):
            if dict_count['train'][label] >= N:
                break
            dict_count['train'][label] += 1
            img = Image.open(f'{data_path}/{phase}/{label}/{file_name}')
            new_img = ImageOps.flip(img)
            new_img.save(f'{data_path}/{phase}/{label}/{file_name.split(".")[0]}_flip_regular.png')


def image_corners(data_path='data', k=3):
    N = 850
    print("Augmenting with different locations")
    dic = stats()
    phase_path = f'{data_path}/train'
    for label in os.listdir(phase_path):
        n = max(0, (N - dic['train'][label])) // k
        label_path = f'{phase_path}/{label}'
        for image_name in os.listdir(label_path)[:n]:
            img_path = f'{label_path}/{image_name}'
            img = Image.open(img_path)
            tl, tr, bl, br, center = corners_and_center(img)
            for idx, aug in enumerate(random.sample([tl, tr, bl, br], k=k), start=1):
                aug.save(f"{label_path}/{image_name}_aug{idx}.png")


def augment_combinations(data_path='data', phase='train'):
    print('Augment data with combinations')
    N = 900
    count = 133333
    dict_count = stats(verbose=False)
    for (source1, source2), target in [
        (('i', 'x'), 'ix'),
        (('vii', 'i'), 'viii'),
        (('vi', 'ii'), 'viii'),
        (('v', 'ii'), 'vii'),
        (('v', 'i'), 'vi'),
        (('i', 'v'), 'iv'),
        # (('ii', 'i'), 'iii'),
        # (('i', 'ii'), 'iii'),
        # (('i', 'i'), 'ii'),
    ]:
        images_sources1 = os.listdir(f'{data_path}/{phase}/{source1}')
        images_sources2 = os.listdir(f'{data_path}/{phase}/{source2}')
        for img1_path in images_sources1:
            for img2_path in images_sources2[1:]:
                if dict_count['train'][target] >= N:
                    break
                dict_count['train'][target] += 1
                img1 = Image.open(f'{data_path}/{phase}/{source1}/{img1_path}')
                img2 = Image.open(f'{data_path}/{phase}/{source2}/{img2_path}')
                combine_images(img1, img2).save(f"{data_path}/{phase}/{target}/combinations_{count}.png")
                count += 1


def blur_augmentation(data_path='data'):
    print("Augmentation with blurring")
    N = 960
    dict_count = stats(verbose=False)
    count = 100000
    for target in [key for key, value in stats(verbose=False)['train'].items() if value < N]:
        images = os.listdir(f'{data_path}/train/{target}')
        for fname in images[:max(0, N - dict_count['train'][target])]:
            count += 1
            img = Image.open(f'{data_path}/train/{target}/{fname}')
            img.save(f'{data_path}/train/{target}/{count}_blurred.png')


def rotate_augmentation(data_path='data'):
    global N
    print("Rotate Augmentation")
    dict_count = stats(verbose=False)
    count = 302922
    for target in [key for key, value in stats(verbose=False)['train'].items() if value < N]:
        images = os.listdir(f'{data_path}/train/{target}')
        for fname in images[:max(0, N - dict_count['train'][target])]:
            count += 1
            img = Image.open(f'{data_path}/train/{target}/{fname}')
            sign = (random.randint(0, 1) * 2) - 1
            img.rotate(sign * 15, fillcolor=255)
            img.save(f'{data_path}/train/{target}/{count}_blurred.png')
    dict_count = stats(verbose=False)
