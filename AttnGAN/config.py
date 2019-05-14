import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
'''
general config
'''
parser.add_argument('--seed', type=int, default=7412)

'''
data
'''
parser.add_argument('--data_dir', type=str, default='../../data/COCO')
parser.add_argument('--words_num', type=int, default=15)

'''
encoders
'''
parser.add_argument('--image_encoder_path', type=str, default='')
parser.add_argument('--text_encoder_path', type=str, default='')
parser.add_argument('--t_dim', type=int, default=256)

'''
gan networks
'''
parser.add_argument('--branch_num', type=int, default=3)
parser.add_argument('--base_size', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--c_dim', type=int, default=100)
parser.add_argument('--ndf', type=int, default=96)
parser.add_argument('--ngf', type=int, default=48)

'''
optimizers
'''
parser.add_argument('--lrD', type=float, default=0.0002)
parser.add_argument('--lrG', type=float, default=0.0002)

'''
train
'''
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--snapshot_interval', type=int, default=5)

'''
loss
'''
parser.add_argument('--gamma1', type=float, default=4.0)
parser.add_argument('--gamma2', type=float, default=5.0)
parser.add_argument('--gamma3', type=float, default=10.0)
parser.add_argument('--smooth_lambda', type=float, default=50.0)


config = parser.parse_args()
config.date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
