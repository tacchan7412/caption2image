import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

from global_attention import GlobalAttentionGeneral as f_attn


'''
utility blocks
'''


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


def Block3x3_leakyReLU(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


'''
Encoders
'''


class RNNEncoder(nn.Module):
    def __init__(self, ntoken, words_num, rnn_type='LSTM',
                 ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNNEncoder, self).__init__()
        self.n_steps = words_num
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, captions, cap_lens):
        # input: torch.LongTensor of size batch x n_steps
        emb = self.drop(self.encoder(captions))
        # emb: batch x n_steps x ninput

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # emb: batch x n_steps x ninput

        # initial hidden state is not provided,
        # resulting in zero initializations of the state
        output, hidden = self.rnn(emb)
        # output: batch x n_steps x num_directions * num_hidden
        # hidden: num_layers * num_directions x batch x num_hidden
        # hidden = (hidden state, cell state)

        output = pad_packed_sequence(output, batch_first=True)[0]
        # output: batch x n_steps x num_directions * num_hidden

        words_emb = output.transpose(1, 2)
        # word_emb:  batch x num_directions * num_hidden x n_steps
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, (self.nlayers *
                                      self.nhidden *
                                      self.num_directions))
        # sent_emb: batch x num_layers * num_directions * num_hidden
        return words_emb, sent_emb


class CNNEncoder(nn.Module):
    def __init__(self, nef):
        super(CNNEncoder, self).__init__()
        self.nef = nef

        self.define_module()
        self.init_trainable_weights()

    def define_module(self):
        model = models.inception_v3(pretrained=True)
        # set requires_grad False for inception v3
        for param in model.parameters():
            param.requires_grad = False
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # below is as same as inception v3
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        # image region features
        features = x
        # features: batch x 17 x 17 x 768

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        # x: batch x 2048

        cnn_code = self.emb_cnn_code(x)
        features = self.emb_features(features)
        return features, cnn_code


'''
Generator
'''


class CANet(nn.Module):
    def __init__(self, t_dim, c_dim, device):
        super(CANet, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU(True)
        self.device = device

    def encode(self, sent_emb):
        x = self.relu(self.fc(sent_emb))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps.mul(std).add(mu)

    def forward(self, sent_emb):
        mu, logvar = self.encode(sent_emb)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class F0Net(nn.Module):
    '''
    h0 = F0(z, F_ca(e))
    z: batch x z_dim
    F_ca(e): batch x c_dim
    h0: batch x ngf/16 x 64 x 64
    '''
    def __init__(self, ngf, z_dim, c_dim):
        super(F0Net, self).__init__()
        self.c_dim = c_dim
        self.ngf = ngf
        self.in_dim = z_dim + c_dim

        self.define_module()

    def define_module(self):
        ngf = self.ngf
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z, c_code):
        z_c_code = torch.cat([z, c_code], 1)
        h0 = self.fc(z_c_code).view(-1, self.ngf, 4, 4)
        h0 = self.upsample1(h0)
        h0 = self.upsample2(h0)
        h0 = self.upsample3(h0)
        h0 = self.upsample4(h0)
        return h0


class FNet(nn.Module):
    '''
    h_i = F_i(h_{i-1}, F_attn_i(e, h_{i-1}))
    h_{i-1}: batch x ngf x h x w
    e: batch x nef x n_step
    F_attn_i(e, h_{i-1}): batch x ngf x h x w
    h_i: batch x ngf x 2*h x 2*w
    '''
    def __init__(self, ngf, nef, num_res=2):
        super(FNet, self).__init__()
        self.ngf = ngf
        self.nef = nef
        self.num_res = num_res

        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for _ in range(self.num_res):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.ngf
        self.f_attn = f_attn(ngf, self.nef)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, word_embs, mask):
        self.f_attn.applyMask(mask)
        c_code, attn = self.f_attn(h_code, word_embs)
        # channel concat
        h_c_code = torch.cat([h_code, c_code], 1)
        out = self.residual(h_c_code)
        out = self.upsample(out)
        return out, attn


class GNet(nn.Module):
    '''
    x_hat_i = G_i(h_i)
    h_i: batch x ngf x h x w
    x_hat_i: batch x 3 x h x w
    '''
    def __init__(self, ngf):
        super(GNet, self).__init__()
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)


class AttentionalGNet(nn.Module):
    def __init__(self, z_dim, t_dim, c_dim, ngf,
                 device, branch_num=3):
        super(AttentionalGNet, self).__init__()
        self.branch_num = branch_num

        self.ca_net = CANet(t_dim, c_dim, device)
        if branch_num > 0:
            self.f0 = F0Net(ngf * 16, z_dim, c_dim)
            self.g0 = GNet(ngf)
        if branch_num > 1:
            self.f1 = FNet(ngf, t_dim)
            self.g1 = GNet(ngf)
        if branch_num > 2:
            self.f2 = FNet(ngf, t_dim)
            self.g2 = GNet(ngf)

    def forward(self, z, sent_emb, word_embs, mask):
        fake_imgs = []
        attn_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if self.branch_num > 0:
            h_code = self.f0(z, c_code)
            fake_img = self.g0(h_code)
            fake_imgs.append(fake_img)
        if self.branch_num > 1:
            h_code, attn = self.f1(h_code, word_embs, mask)
            fake_img = self.g1(h_code)
            fake_imgs.append(fake_img)
            if attn is not None:
                attn_maps.append(attn)
        if self.branch_num > 2:
            h_code, attn = self.f2(h_code, word_embs, mask)
            fake_img = self.g2(h_code)
            fake_imgs.append(fake_img)
            if attn is not None:
                attn_maps.append(attn)
        return fake_imgs, attn_maps, mu, logvar


'''
Discriminator
'''


class DLogit(nn.Module):
    def __init__(self, ndf, t_dim, cond=False):
        super(DLogit, self).__init__()
        self.ndf = ndf
        self.t_dim = t_dim
        self.cond = cond

        if cond:
            self.jointConv = Block3x3_leakyReLU(ndf * 8 + t_dim, ndf * 8)

        self.main = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.cond and c_code is not None:
            c_code = c_code.view(-1, self.t_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat([h_code, c_code], 1)
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code
        out = self.main(h_c_code)
        return out.view(-1)


class DNet64(nn.Module):
    def __init__(self, ndf, t_dim, jcu=True):
        super(DNet64, self).__init__()
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.uncond_dnet = None
        if jcu:
            self.uncond_dnet = DLogit(ndf, t_dim, cond=False)
        self.cond_dnet = DLogit(ndf, t_dim, cond=True)

    def forward(self, x):
        return self.img_code_s16(x)  # batch x 8ndf x 4 x 4


class DNet128(nn.Module):
    def __init__(self, ndf, t_dim, jcu=True):
        super(DNet128, self).__init__()
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakyReLU(ndf * 16, ndf * 8)
        self.uncond_dnet = None
        if jcu:
            self.uncond_dnet = DLogit(ndf, t_dim, cond=False)
        self.cond_dnet = DLogit(ndf, t_dim, cond=True)

    def forward(self, x):
        x = self.img_code_s16(x)
        x = self.img_code_s32(x)
        x = self.img_code_s32_1(x)
        return x  # batch x 8ndf x 4 x 4


class DNet256(nn.Module):
    def __init__(self, ndf, t_dim, jcu=True):
        super(DNet256, self).__init__()
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakyReLU(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakyReLU(ndf * 16, ndf * 8)
        self.uncond_dnet = None
        if jcu:
            self.uncond_dnet = DLogit(ndf, t_dim, cond=False)
        self.cond_dnet = DLogit(ndf, t_dim, cond=True)

    def forward(self, x):
        x = self.img_code_s16(x)
        x = self.img_code_s32(x)
        x = self.img_code_s64(x)
        x = self.img_code_s64_1(x)
        x = self.img_code_s64_2(x)
        return x  # batch x 8ndf x 4 x 4
