from collections import defaultdict
from tqdm import tqdm
import os

import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def stats(data_path='data', verbose=False) -> dict:
    dict_count = {}
    for phase in ['train', 'val']:
        dict_count[phase] = {}
        count = 0
        for class_name in os.listdir(f'{data_path}/{phase}'):
            if verbose:
                print(f'{phase}\t{class_name}\tN={len(os.listdir(f"{data_path}/{phase}/{class_name}"))}')
            count += len(os.listdir(f"{data_path}/{phase}/{class_name}"))
            dict_count[phase][class_name] = len(os.listdir(f"{data_path}/{phase}/{class_name}"))
        if verbose:
            print(f'{phase}\t{count} total images')
    return dict_count


def vis(model, val_dataloader, plot_wrong_label="NONE") -> defaultdict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = val_dataloader.dataset.classes
    wrongs = defaultdict(int)
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(1)
            for img, y, y_hat in list(zip(inputs[~(labels == pred)], labels[~(labels == pred)], pred[~(labels == pred)])):
                wrongs[(classes[y.item()], classes[y_hat.item()])] += 1
                plt.figure()
                plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
                plt.title(f'True = {classes[y.item()]}    Wrong = {classes[y_hat.item()]}')
                plt.show()
    return wrongs


def plot_pca(components, labels):
    import plotly.express as px
    # fig.update_layout(title_text='PCA Dimension Reduction - Cleane Augmented Images', title_x=0.5)
    fig = px.scatter(components, x=0, y=1, color=labels)
    fig.update_layout(title_text='PCA Dimension Reduction - After Augmentation', title_x=0.5)
    # fig.update_xaxes(scaleanchor = "x", scaleratio = 1,)
    fig.update_xaxes(range=(-50, 150), constrain='domain')
    fig.update_yaxes(range=(-30, 40), constrain='domain')
    fig.show()
