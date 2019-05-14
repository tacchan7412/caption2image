import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import time
import random
from datetime import datetime
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

from datasets import TextDataset
from datasets import prepare_data
from models import RNNEncoder
from models import CNNEncoder
from losses import sent_loss
from losses import words_loss
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# config
imsize = 299
batch_size = 48
lr = 0.0002
epochs = 600
seed = 7412
snapshot_interval = 5

words_num = 15
embedding_dim = 256
rnn_grad_clip = 0.25

gamma1 = 4.0
gamma2 = 5.0
gamma3 = 10.0


def train(dataloader, cnn_model, rnn_model, batch_size, words_num,
          labels, optimizer, iters, writer, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()

    for i, data in enumerate(dataloader):
        iters += 1
        optimizer.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data, device)

        words_features, sent_code = cnn_model(imgs[-1])
        att_sze = words_features.size(2)
        words_emb, sent_emb = rnn_model(captions, cap_lens)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb,
                                                 labels, cap_lens, class_ids,
                                                 batch_size, device,
                                                 gamma1, gamma2, gamma3)
        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels,
                                     class_ids, batch_size,
                                     device, gamma3)
        loss = w_loss0 + w_loss1 + s_loss0 + s_loss1
        writer.add_scalar('loss', loss.item(), iters)
        scalars_dic = {'w_loss0': w_loss0.item(),
                       'w_loss1': w_loss1.item(),
                       's_loss0': s_loss0.item(),
                       's_loss1': s_loss1.item()}
        writer.add_scalars('losses', scalars_dic, iters)
        loss.backward()

        torch.nn.utils.clip_grad_norm(rnn_model.parameters(), rnn_grad_clip)
        optimizer.step()

        if iters % 200 == 0:
            print('[%6d]\tw_loss: (%.4f, %.4f)\ts_loss: (%.4f, %.4f)'
                  % (iters, w_loss0.item(), w_loss1.item(),
                     s_loss0.item(), s_loss1.item()))

            # attention Maps
            img_set, _ = \
                utils.build_super_images(imgs[-1].cpu(), captions,
                                         ixtoword, attn_maps, att_sze,
                                         batch_size, words_num)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, iters)
                im.save(fullpath)

    return iters


def evaluate(dataloader, cnn_model, rnn_model, batch_size, iters):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    cnt = 0
    for data in dataloader:
        cnt += 1
        real_imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data, device)

        words_features, sent_code = cnn_model(real_imgs[-1])

        words_emb, sent_emb = rnn_model(captions, cap_lens)

        w_loss0, w_loss1, _ = words_loss(words_features, words_emb, labels,
                                         cap_lens, class_ids, batch_size,
                                         device, gamma1, gamma2, gamma3)
        w_total_loss += w_loss0.item() + w_loss1.item()

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size,
                      device, gamma3)
        s_total_loss += s_loss0.item() + s_loss1.item()

        if cnt == 50:
            break

    s_cur_loss = s_total_loss / cnt
    w_cur_loss = w_total_loss / cnt
    cur_loss = s_cur_loss + w_cur_loss
    writer.add_scalar('val_loss', cur_loss, iters)
    scalars_dic = {'w_loss': w_cur_loss,
                   's_loss': s_cur_loss}
    writer.add_scalars('val_losses', scalars_dic, iters)

    return s_cur_loss, w_cur_loss


if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
    result_dir = os.path.join('results/DAMSM', 'COCO', date_str)
    utils.mkdir_p(result_dir)
    image_dir = os.path.join(result_dir, 'images')
    utils.mkdir_p(image_dir)
    model_dir = os.path.join(result_dir, 'model')
    utils.mkdir_p(model_dir)

    writer = SummaryWriter('runs/DAMSM/COCO/' + date_str)

    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset('../../data/COCO', 'train2014',
                          base_size=imsize, branch_num=1,
                          words_num=words_num,
                          transform=image_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=4)
    dataset_val = TextDataset('../../data/COCO', 'val2014',
                              base_size=imsize, branch_num=1,
                              words_num=words_num,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=4)

    text_encoder = RNNEncoder(dataset.n_words, words_num,
                              nhidden=embedding_dim).to(device)
    image_encoder = CNNEncoder(embedding_dim).to(device)
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        text_encoder = nn.DataParallel(text_encoder)
        image_encoder = nn.DataParallel(image_encoder)

    labels = torch.LongTensor(range(batch_size)).to(device)

    text_encoder_params = text_encoder.parameters()
    image_encoder_params = \
        filter(lambda p: p.requires_grad, image_encoder.parameters())

    iters = 0
    lr_copy = lr

    try:
        for epoch in range(1, epochs+1):
            optimizer = optim.Adam([{'params': text_encoder_params},
                                    {'params': image_encoder_params}],
                                   lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            iters = train(dataloader, image_encoder, text_encoder,
                          batch_size, words_num, labels, optimizer,
                          iters, writer, dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size, iters)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > lr_copy/10.:
                lr *= 0.98

            if (epoch % snapshot_interval == 0 or epoch == epochs):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
