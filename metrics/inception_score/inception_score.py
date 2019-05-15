import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, device, batch_size=32, resize=True, splits=1):
    """
    computes inception score of given images (imgs)
    imgs: numpy images (NxncxHxW) normalized in the range [0, 1]
    batch_size:batch size for feeding into Inception v3
    splits: number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # preprocess imgs
    # grayscale to RGB
    if len(imgs.shape) == 4 and imgs.shape[1] == 1:
        imgs = np.repeat(imgs, 3, axis=1)
    elif len(imgs.shape) == 3:
        imgs = np.repeat(imgs.reshape((N, 1, imgs.shape[1], imgs.shape[2])),
                         3, axis=1)
    # normalize as input of Inception v3
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear')

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            batch_size_i = batch.size(0)
            batch = batch.type(torch.FloatTensor).to(device)

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
