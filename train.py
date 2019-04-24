import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

import os
import random
from datetime import datetime

import matplotlib.pyplot as plt

import utils
from dataset import COCODataset
from model import Generator
from model import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


epochs = 100
batch_size = 64
seed = 7412
nc = 3
nte = 1024
nt = 256
nz = 100
ngf = 1024
ndf = 64
lr = 0.0002
beta1 = 0.5

cls_flag = True
int_flag = True
int_beta = 0.5

gan_name = 'gan'
if int_flag:
    gan_name += '_int'
if cls_flag:
    gan_name += '_cls'

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
result_dir = os.path.join('results',
                          'COCO',
                          gan_name,
                          date_str)
if not os.path.exists(result_dir):
        os.makedirs(result_dir)

num_test_images = 100
fixed_z = torch.randn(num_test_images, nz, 1, 1).to(device)
fixed_ts = torch.FloatTensor(num_test_images, nte).to(device)

real, fake = 1, 0

G_losses = []
D_losses = []


def train(epoch, G_net, D_net, dataloader, G_opt, D_opt, criterion):
    G_net.train()
    D_net.train()

    real_label = torch.FloatTensor(batch_size, 1).fill_(real).to(device)
    fake_label = torch.FloatTensor(batch_size, 1).fill_(fake).to(device)
    for i, (x, t) in enumerate(dataloader):
        x = x[:batch_size].to(device)
        t, t_unmatch = t[:batch_size].to(device), t[batch_size:].to(device)
        z = torch.randn(batch_size, nz, 1, 1).to(device)

        # update Discriminator
        D_opt.zero_grad()
        D_real = D_net(x, t).view(-1)
        D_real_loss = criterion(D_real, real_label)
        D_real_loss.backward()
        D_loss = D_real_loss
        if cls_flag:
            D_fake_text = D_net(x, t_unmatch).view(-1)
            D_fake_text_loss = criterion(D_fake_text, fake_label) * 0.5
            D_fake_text_loss.backward()
            D_loss += D_fake_text_loss
        x_hat = G_net(z, t)
        D_fake_image = D_net(x_hat.detach(), t).view(-1)
        D_fake_image_loss = criterion(D_fake_image, fake_label) * (1.0 - 0.5 * cls_flag)
        D_fake_image_loss.backward()
        D_loss += D_fake_image_loss
        D_opt.step()

        # update Generator
        G_opt.zero_grad()
        D_fake = D_net(x_hat, t).view(-1)
        G_fake_loss = criterion(D_fake, real_label)
        G_fake_loss.backward()
        G_loss = G_fake_loss
        if int_flag:
            t_int = int_beta * t + (1 - int_beta) * t_unmatch
            D_fake_int = D_net(G_net(z, t_int), t_int).view(-1)
            G_fake_int_loss = criterion(D_fake_int, real_label)
            G_fake_int_loss.backward()
            G_loss += G_fake_int_loss
        G_opt.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, epochs, i, len(dataloader), D_loss.item(), G_loss.item()))
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())


def prepare_test(dataloader):
    ans_images = torch.FloatTensor(num_test_images, 3, 64, 64)
    save_dir = os.path.join(result_dir, 'images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start, end = 0, 0
    for i, (x, t) in enumerate(dataloader):
        t = t.to(device)
        start = end
        end = min((i+1)*batch_size, num_test_images)
        if end == num_test_images:
            ans_images[start:end] = x[:num_test_images % batch_size]
            fixed_ts[start:end] = t[:num_test_images % batch_size]
            break
        else:
            ans_images[start:end] = x
            fixed_ts[start:end] = t

    vutils.save_image(ans_images,
                      os.path.join(save_dir, 'answer.png'),
                      nrow=10)


def test(epoch, G_net):
    save_dir = os.path.join(result_dir, 'images')
    G_net.eval()
    with torch.no_grad():
        images = G_net(fixed_z, fixed_ts)
        vutils.save_image(images,
                          os.path.join(save_dir, 'epoch%03d.png' % (epoch)),
                          nrow=10)


def save(epoch, G_net, D_net):
    save_dir = os.path.join(result_dir, 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(G_net.state_dict(),
               os.path.join(save_dir, 'G_epoch%03d.pkl' % (epoch)))
    torch.save(D_net.state_dict(),
               os.path.join(save_dir, 'D_epoch%03d.pkl' % (epoch)))


def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'loss_plot.png'))
    plt.close()


def main():
    # networks
    G_net = Generator(nc, nz, nte, nt, ngf).to(device)
    D_net = Discriminator(nc, nte, nt, ndf).to(device)
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        G_net = nn.DataParallel(G_net)
        D_net = nn.DataParallel(D_net)
    G_net.apply(utils.weights_init)
    D_net.apply(utils.weights_init)
    print(G_net)
    print(D_net)

    # criterion
    criterion = nn.BCELoss()

    # optimizers
    G_opt = optim.Adam(G_net.parameters(), lr=lr, betas=(beta1, 0.999))
    D_opt = optim.Adam(D_net.parameters(), lr=lr, betas=(beta1, 0.999))

    # dataloader
    image_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))])
    train_dataset = COCODataset('../data/COCO/train2014', transform=image_transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size*2,  # half of batch will be used for fake text
        shuffle=True,
        drop_last=True,
        num_workers=4)
    dummy_train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    print('prepare embeddings for test')
    prepare_test(dummy_train_dataloader)
    del dummy_train_dataloader

    print('start train')
    for epoch in range(1, epochs+1):
        train(epoch, G_net, D_net, train_dataloader, G_opt, D_opt, criterion)
        test(epoch, G_net)
        save(epoch, G_net, D_net)
    plot_losses()


if __name__ == '__main__':
    main()
