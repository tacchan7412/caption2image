import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

import os
import random
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

from config import config as cfg
from datasets import TextDataset
from datasets import prepare_data
from models import CNNEncoder
from models import RNNEncoder
from models import DNet64
from models import DNet128
from models import DNet256
from models import AttentionalGNet
from losses import discriminator_loss
from losses import generator_loss
from losses import KL_loss
from losses import words_loss
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


class AttnGAN(object):
    def __init__(self, dataloader, writer):
        self.dataloader = dataloader
        self.writer = writer
        self.iter = 0

        result_dir = os.path.join('results', 'AttnGAN', 'COCO', cfg.date_str)
        utils.mkdir_p(result_dir)
        # save cfg value to txt file
        txt_path = os.path.join(result_dir, 'config.txt')
        with open(txt_path, 'w') as f:
            for k, v in vars(cfg).items():
                text = k + ': ' + str(v)
                print(text, file=f)

        self.image_dir = os.path.join(result_dir, 'images')
        utils.mkdir_p(self.image_dir)
        self.model_dir = os.path.join(result_dir, 'model')
        utils.mkdir_p(self.model_dir)

        self.dataset = self.dataloader.dataset
        self.ixtoword = self.dataset.ixtoword

    def build_models(self):
        image_encoder = CNNEncoder(cfg.t_dim).to(device)
        image_encoder.load_state_dict(torch.load(cfg.image_encoder_path))
        for p in image_encoder.parameters():
            p.requires_grad = False

        text_encoder = RNNEncoder(self.dataset.n_words,
                                  cfg.words_num, nhidden=cfg.t_dim).to(device)
        text_encoder.load_state_dict(torch.load(cfg.text_encoder_path))
        for p in text_encoder.parameters():
            p.requires_grad = False

        Ds = []
        if cfg.branch_num > 0:
            Ds.append(DNet64(cfg.ndf, cfg.t_dim).to(device))
        if cfg.branch_num > 1:
            Ds.append(DNet128(cfg.ndf, cfg.t_dim).to(device))
        if cfg.branch_num > 2:
            Ds.append(DNet256(cfg.ndf, cfg.t_dim).to(device))
        for D in Ds:
            D.apply(utils.weights_init)
        G = AttentionalGNet(cfg.z_dim, cfg.t_dim, cfg.c_dim, cfg.ngf,
                            device, cfg.branch_num).to(device)
        G.apply(utils.weights_init)
        # TODO: load G/D from path
        if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
            for D in Ds:
                D = nn.DataParallel(D)
            G = nn.DataParallel(G)
        return [text_encoder, image_encoder, Ds, G]

    def define_optimizers(self, G, Ds):
        D_optimizers = []
        for D in Ds:
            opt = optim.Adam(D.parameters(),
                             lr=cfg.lrD, betas=(0.5, 0.999))
            D_optimizers.append(opt)
        G_optimizer = optim.Adam(G.parameters(),
                                 lr=cfg.lrG, betas=(0.5, 0.999))
        return G_optimizer, D_optimizers

    def prepare_labels(self):
        batch_size = cfg.batch_size
        real_labels = torch.FloatTensor(batch_size).fill_(1).to(device)
        fake_labels = torch.FloatTensor(batch_size).fill_(0).to(device)
        match_labels = torch.LongTensor(range(batch_size)).to(device)
        return real_labels, fake_labels, match_labels

    def prepare_fixed_data(self, text_encoder):
        noise = torch.FloatTensor(cfg.batch_size, cfg.z_dim).normal_()
        noise = noise.to(device)
        data_iter = iter(self.dataloader)
        data = data_iter.next()
        imgs, captions, cap_lens, class_ids, keys = \
            prepare_data(data, device)
        words_embs, sent_emb = text_encoder(captions, cap_lens)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        return (noise, sent_emb, words_embs, mask, captions, cap_lens)

    def save_models(self, G, Ds, epoch):
        torch.save(G.state_dict(),
                   os.path.join(self.model_dir, 'G_epoch%d.pth' % (epoch)))
        for i, D in enumerate(Ds):
            torch.save(D.state_dict(),
                       os.path.join(self.model_dir,
                                    'D_epoch%d_%d.pth' % (epoch, i)))
        print('saved G/D models on epoch', epoch)

    def save_images(self, G, noise, sent_emb, words_embs, mask,
                    image_encoder, captions, cap_lens, epoch):
        G.eval()
        with torch.no_grad():
            fake_imgs, attention_maps, _, _ = G(noise, sent_emb,
                                                words_embs, mask)
        for i, fake_img in enumerate(fake_imgs):
            img = vutils.make_grid(fake_img.detach())
            self.writer.add_image('images/%d' % (i), img, epoch)

        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                utils.build_super_images(img, captions, self.ixtoword,
                                         attn_maps, att_sze,
                                         cfg.batch_size, cfg.words_num,
                                         lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%d_%d.png'\
                    % (self.image_dir, epoch, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, cfg.batch_size,
                                    device, cfg.gamma1, cfg.gamma2, cfg.gamma3)
        img_set, _ = \
            utils.build_super_images(fake_imgs[i].detach().cpu(),
                                     captions, self.ixtoword,
                                     att_maps, att_sze,
                                     cfg.batch_size, cfg.words_num)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%d.png'\
                % (self.image_dir, epoch)
            im.save(fullpath)
        print('saved images on epoch', epoch)

    def train(self):
        text_encoder, image_encoder, Ds, G = self.build_models()
        G_optimizer, D_optimizers = self.define_optimizers(G, Ds)
        real_labels, fake_labels, match_labels = self.prepare_labels()
        noise = torch.FloatTensor(cfg.batch_size, cfg.z_dim).to(device)
        (fixed_noise, fixed_sent_emb, fixed_words_embs,
         fixed_mask, fixed_captions, fixed_cap_lens) = \
            self.prepare_fixed_data(text_encoder)

        for epoch in range(1, cfg.epochs+1):
            print('start epoch', epoch)
            G.train()
            for D in Ds:
                D.train()
            for i, data in enumerate(self.dataloader):
                self.iter += 1
                imgs, captions, cap_lens, class_ids, keys = \
                    prepare_data(data, device)
                words_embs, sent_emb = text_encoder(captions, cap_lens)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                noise.data.normal_()
                fake_imgs, _, mu, logvar = G(noise, sent_emb, words_embs, mask)

                # update Ds
                for i, D in enumerate(Ds):
                    D.zero_grad()
                    errD, scalars_dic = \
                        discriminator_loss(D, imgs[i], fake_imgs[i],
                                           sent_emb, real_labels,
                                           fake_labels)
                    self.writer.add_scalar('D%d_loss' % (i), errD.item(),
                                           self.iter)
                    self.writer.add_scalars('D%d_loss_detail' % (i),
                                            scalars_dic, self.iter)
                    errD.backward()
                    D_optimizers[i].step()

                # update G
                G.zero_grad()
                errG, scalars_dic = \
                    generator_loss(Ds, image_encoder, fake_imgs,
                                   real_labels, words_embs, sent_emb,
                                   match_labels, cap_lens, class_ids,
                                   device, cfg.gamma1, cfg.gamma2,
                                   cfg.gamma3, cfg.smooth_lambda)
                kl_loss = KL_loss(mu, logvar)
                scalars_dic['kl_loss'] = kl_loss.item()
                errG += kl_loss
                self.writer.add_scalar('G_loss', errG.item(), self.iter)
                self.writer.add_scalars('G_loss_detail', scalars_dic,
                                        self.iter)
                errG.backward()
                G_optimizer.step()
                if self.iter % 100 == 0:
                    print('iter', self.iter)

            self.save_images(G, fixed_noise, fixed_sent_emb, fixed_words_embs,
                             fixed_mask, image_encoder, fixed_captions,
                             fixed_cap_lens, epoch)
            if (epoch % cfg.snapshot_interval == 0 or epoch == cfg.epochs):
                self.save_models(G, Ds, epoch)


if __name__ == '__main__':
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    writer = SummaryWriter('runs/AttnGAN/COCO/' + cfg.date_str)

    imsize = cfg.base_size * (2 ** (cfg.branch_num - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.data_dir, 'train2014',
                          base_size=cfg.base_size, branch_num=cfg.branch_num,
                          words_num=cfg.words_num,
                          transform=image_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, drop_last=True,
        shuffle=True, num_workers=4)

    gan = AttnGAN(dataloader, writer)
    gan.train()
