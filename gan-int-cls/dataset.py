import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np


class COCODataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 imsize=64, transform=None):

        self.transform = transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir

        self.filenames = self.load_filenames()
        self.embeddings = self.load_embedding(embedding_type)
        # self.captions = self.load_all_captions()

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_all_captions(self):
        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = self.load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_captions(self, caption_name):
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().decode('utf8').split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions

    def load_embedding(self, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '../../data/COCO/coco/train/char-CNN-RNN-embeddings.pickle'

        with open(embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_filenames(self):
        filepath = os.path.join('../../data/COCO/coco/train', 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]

        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/%s.jpg' % (self.data_dir, key)
        img = self.get_img(img_name)

        rand_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[rand_ix, :]
        return img, embedding  #, captions[rand_ix]

    def __len__(self):
        return len(self.filenames)
