import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nc, nz, nte, nt, ngf):
        super(Generator, self).__init__()
        self.nt = nt
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz+nt, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            # (ngf*8) x 4 x 4
            nn.Conv2d(ngf*8, ngf*2, 1, 1, 0),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*2, 3, 1, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*8, 3, 1, 1),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            # (ngf*4) x 8 x 8
            nn.Conv2d(ngf*4, ngf, 1, 1, 0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf*4, 3, 1, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # nc x 64 x 64
        )
        self.enc_text = nn.Sequential(
            nn.Linear(nte, nt),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, z, t):
        enc_text = self.enc_text(t).view(-1, self.nt, 1, 1)
        zt = torch.cat([z, enc_text], 1)
        out = self.main(zt)
        return out


class Discriminator(nn.Module):
    def __init__(self, nc, nte, nt, ndf):
        super(Discriminator, self).__init__()
        self.nt = nt
        self.conv1 = nn.Sequential(
            # nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.Conv2d(ndf*8, ndf*2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True)
            # (ndf*8) x 4 x 4
        )
        self.enc_text = nn.Sequential(
            nn.Linear(nte, nt),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf*8+nt, ndf*8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        h = self.conv1(x)
        enc_text = self.enc_text(t).view(-1, self.nt, 1, 1).repeat(1, 1, 4, 4)
        out = self.conv2(torch.cat([h, enc_text], 1))
        return out.view(-1, 1).squeeze(1)
