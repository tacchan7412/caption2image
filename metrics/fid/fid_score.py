import torch
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from metrics.fid.inception import InceptionV3


class fid(object):
    def __init__(self, device, dims=2048):
        """
        dataloader: torch.utils.data.Dataloader
                    calc m1 and s1 first and reuse
        """
        self.device = device

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)

    def calculate_score(self, real_imgs, imgs, batch_size=32):
        """
        imgs: numpy images (NxncxHxW) normalized in the range [0, 1]
        """
        dataloader = torch.utils.data.DataLoader(real_imgs,
                                                 batch_size=batch_size)
        m1, s1 = self.calculate_activation_statistics(dataloader, self.model)
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
        m2, s2 = self.calculate_activation_statistics(dataloader, self.model)
        fid_value = \
            self.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_activation_statistics(self, dataloader, model, dims=2048):
        act = self.get_activations(dataloader, model, dims)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def get_activations(self, dataloader, model, dims=2048):
        """
        Calculates the activations of the pool_3 layer
        for all images in dataloader
        image in dataloader has shape of NxncxHxW and value in [0, 1]
        """
        model.eval()

        pred_arr = np.empty((len(dataloader.dataset), dims))
        start, end = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # for shorter calculation time
                if end > 4000:
                    continue
                if isinstance(batch, list):
                    # the case which data is (x, y)
                    batch = batch[0]
                batch_size_i = batch.size(0)
                start = end
                end = start + batch_size_i

                # make data shape to Nx3xHxW
                if len(batch.size()) == 4 and batch.size(1) == 1:
                    batch = batch.repeat(1, 3, 1, 1)
                elif len(batch.size()) == 3:
                    batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

                batch = batch.type(torch.FloatTensor).to(self.device)
                pred = model(batch)[0]

                # If model output is not scalar,
                # apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                pred_arr[start:end] = \
                    pred.cpu().data.numpy().reshape(batch_size_i, -1)

        return pred_arr[:end]
