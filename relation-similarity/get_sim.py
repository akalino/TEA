import argparse
import json
import logging
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from comparison_utils import make_diagonal, normalize, von_neumann_entropy
from models.baseModelOrig import baseModel
from util.dataloader import relationDataset

args = argparse.ArgumentParser()
args.add_argument('--model_path', required=True, help='path to directory that contains model file')
args.add_argument('--dataset', required=True, help='The name of the dataset to use')
args.add_argument('--class', required=True, help='The name of the entity embedding model class')
args.add_argument('--input', required=True, help='path to directory that contains input files')
args.add_argument('--output', required=True, help='path to directory that output files will be stored')
args.add_argument('--sample_num', type=int, default=20, help='how many samples are used when approximating KL divergence')
args.add_argument('--gpu', default='0', help='ID of the gpu you want to assign')
args = vars(args.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(message)s')

if not os.path.exists(args['output']):
    os.mkdir(args['output'])


with open(os.path.join(args['model_path'], '{}-{}-info.json'.format(args['dataset'], args['class'])), 'r') as f:
    info = json.load(f)
logging.info("Loading model")
model = baseModel(info, info['tot_ent'], info['tot_rel'])
flag = False
model.load_state_dict(torch.load(os.path.join(args['model_path'],
                                              '{}-{}-model.pth'.format(args['dataset'], args['class']))))
flag = True
if not flag:
    raise IOError('Model file not found.')

model.cuda()

ds_name = args['input'].split('/')
dn = ds_name[-2]

dataset = relationDataset(os.path.join(args['input'], 'train.txt'),
                          os.path.join(args['input'], 'entity2id.txt'),
                          os.path.join(args['input'], 'relation2id.txt'))
logging.info("Calculating similarity")
scores = []
for i in tqdm(range(model.tot_rel)):
    score = model.eval_step(i, args['sample_num'])
    scores.append((score/args['sample_num']).tolist())

scores_tensor = torch.Tensor(scores)
scores = torch.max(scores_tensor, scores_tensor.transpose(1, 0)).tolist()

with open(os.path.join(args['output'], "kl_prob_{}_{}_{}.json".format(args['dataset'],
                                                                      args['class'],
                                                                      args['sample_num'])), 'w') as f:
    json.dump(scores, f)

with open(os.path.join(args['output'], 'kl_prob_{}_{}_{}.txt'.format(args['dataset'],
                                                                     args['class'],
                                                                     args['sample_num'])), 'w') as f:
    for idx1, item in enumerate(scores):
        for idx2, num in enumerate(item):
            if idx1 == idx2:
                continue
            f.write(str(num) + ' ')
        f.write('\n')

_m = np.loadtxt(os.path.join(args['output'], 'kl_prob_{}_{}_{}.txt'.format(args['dataset'],
                                                                           args['class'],
                                                                           args['sample_num'])))

_b = normalize(_m)
_c = make_diagonal(_b, 1.0)
print(_c.shape)
np.save('wnrr-kl.npy', _c)
s_ent = von_neumann_entropy(_c)
print('Fact distribution von Neumann entropy (trace): {}'.format(s_ent))
