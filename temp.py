import torch
import os

ckpt = torch.load('cache.pt')
del ckpt['query_masks']
del ckpt['token_ids']
for i, video in enumerate(ckpt['paths']):
    ckpt['paths'][i] = os.path.basename(str(video))
for i, video in enumerate(ckpt['raw_captions']):
    ckpt['raw_captions'][i] = ' '.join([word for word in ckpt['raw_captions'][i][0]])
torch.save(ckpt, 'cache.pt')