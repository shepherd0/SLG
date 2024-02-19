import argparse
import os


parse = argparse.ArgumentParser(description='PyTorch Cervical cell Segmentation')

"-------------------data option--------------------------"
parse.add_argument('--root', type=str, default='/data/cervix/')
parse.add_argument('--dataset', type=str, default='CX22')
parse.add_argument('--train_data_dir', type=str, default='cervixcx22/train')
parse.add_argument('--valid_data_dir', type=str, default='cervixcx22/valid')
parse.add_argument('--test_data_dir', type=str, default='cervixcx22/test')
parse.add_argument('--input_size', type=int, default=320)

"-------------------training option-----------------------"
parse.add_argument('--mode', type=str, default='train')
parse.add_argument('--nEpoch', type=int, default=200)
parse.add_argument('--batch_size', type=float, default=4)
parse.add_argument('--num_workers', type=int, default=2)
parse.add_argument('--use_gpu', type=bool, default=True)
parse.add_argument('--load_ckpt', type=str, default=None)
parse.add_argument('--model', type=str, default='unet_slg')
parse.add_argument('--expID', type=int, default=0)
parse.add_argument('--ckpt_period', type=int, default=5)

"-------------------optimizer option-----------------------"
parse.add_argument('--lr', type=float, default=1e-3, help="learning rate (default: 0.001)")
parse.add_argument('--weight_decay', type=float, default=1e-5)
parse.add_argument('--mt', type=float, default=0.9)
parse.add_argument('--power', type=float, default=0.9)

parse.add_argument('--nclasses', type=int, default=1)

opt = parse.parse_args()
