import subprocess
import shutil
import os
import random


def renew_data_files():
    """remove data folder and download the new (fixed) one with the data"""
    if os.path.isdir('data'):
        shutil.rmtree('data')

    if os.path.isdir('raw_data'):
        # shutil.rmtree('raw_data')
        return None

    link = "https://technionmail-my.sharepoint.com/:u:/g/personal/eliezer_campus_technion_ac_il/Ee8xO2OrUzlFljYSLBHxYZgBBrU5xnilCbkL6Y265rdgQA?download=1"
    subprocess.Popen(fr"wget {link} -O data.zip".split(), stdout=subprocess.PIPE).communicate()
    subprocess.Popen(r"unzip data".split(), stdout=subprocess.PIPE).communicate()
    subprocess.Popen(r"rm data.zip".split(), stdout=subprocess.PIPE).communicate()
    shutil.move('data_fixed', 'raw_data')


def create_folders():
    augmented_data_path = 'data'
    for class_name in os.listdir('raw_data/train'):
        os.makedirs(f'{augmented_data_path}/train/{class_name}', exist_ok=True)
        os.makedirs(f'{augmented_data_path}/val/{class_name}', exist_ok=True)


def split_train_val(data_path="data", n_images=40):
    # move from train to val some images and some keep in n
    img_idx = 0
    for class_name in os.listdir(f'raw_data/train'):
        files_names = os.listdir(f'raw_data/train/{class_name}')
        random.shuffle(files_names)
        n_images = int(0.2 * (len(files_names) + 1))
        for idx, file_name in enumerate(files_names):
            raw_path = f'raw_data/train/{class_name}/{file_name}'
            path = f'{data_path}/{"val" if idx < n_images else "train"}/{class_name}/image{img_idx}.png'
            shutil.copy(raw_path, path)
            img_idx += 1

    for class_name in os.listdir('raw_data/val'):
        for file_name in os.listdir(f'raw_data/val/{class_name}'):
            shutil.copy(f'raw_data/val/{class_name}/{file_name}', f'{data_path}/val/{class_name}/image{img_idx}.png')
            img_idx += 1
