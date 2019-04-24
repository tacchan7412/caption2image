import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import utils
from dataset import CaptionDataset
from model import Generator
from model import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


epochs = 1
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

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

real, fake = 1, 0

def train(epoch, G_net, D_net, dataloader, G_opt, D_opt, criterion):
    G_net.train()
    D_net.train()

    real_label = torch.FloatTensor(batch_size, 1).fill_(real).to(device)
    fake_label = torch.FloatTensor(batch_size, 1).fill_(fake).to(device)
    for i, (x, t) in enumerate(dataloader):
        x, t = x.to(device), t.to(device)
        z = torch.randn(batch_size, nz, 1, 1).to(device)

        # update Discriminator
        D_opt.zero_grad()
        D_real = D_net(x, t)
        D_real_loss = criterion(D_real, real_label)
        D_real_loss.backward()
        D_fake_text = D_net(torch.cat([x[1:], x[:1]], 0), t)
        D_fake_text_loss = criterion(D_fake_text, fake_label) * 0.5
        D_fake_text_loss.backward()
        x_hat = G_net(z, t)
        D_fake_image = D_net(x_hat.detach(), t)
        D_fake_image_loss = criterion(D_fake_image, fake_label) * 0.5
        D_fake_image_loss.backward()
        D_loss = D_real_loss + D_fake_text_loss + D_fake_image_loss
        D_opt.step()

        # update Generator
        G_opt.zero_grad()
        D_fake = D_net(x_hat, t)
        G_loss = criterion(D_fake, real_label)
        G_loss.backward()
        G_opt.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, epochs, i, len(dataloader), D_loss.item(), G_loss.item()))


def test(epoch, G_net, dataloader):
    pass


def save(epoch, G_net, D_net):
    pass


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

    for epoch in range(1, epochs+1):
        train(epoch, G_net, D_net, dataloader, G_opt, D_opt, criterion)
        test(epoch, G_net, test_dataloader)
        save(epoch, G_net, D_net)


if __name__ == '__main__':
    main()
