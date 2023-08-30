# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-modal Architecture models.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""

import collections
import itertools
import logging
from pickle import NONE
import re
import types

from transformers.modeling_bert import BertLayer,BertConfig, BertSelfAttention, BertIntermediate, BertOutput
from base import BaseModel
from model.VidBert import BertModel
from model.lstm import LSTMModel
from model.net_vlad import NetVLAD
from model.txt_embeddings import TxtEmbeddings
import torch as th
import torch.nn as nn
import torch.nn.functional as F
#from transformers.modeling_bert import BertModel as TxtBertModel
from model.TxtBert import myTxtBertModel

logger = logging.getLogger(__name__)


class EnhancedMMT(BaseModel):
  """Whole cross-modal architecture."""

  def __init__(self,
               l2renorm,
               expert_dims,
               tokenizer,
               keep_missing_modalities,
               test_caption_mode,
               freeze_weights=False,
               mimic_ce_dims=False,
               concat_experts=False,
               concat_mix_experts=False,
               use_experts='origfeat',
               rep_dim=2048,
               proj_dim=8192,
               projector_layer='swlinear',
               embd_out='avg',
               txt_layers='12,1',
               vid_layers='4,1',
               out_space='proj',
               txt_inp=None,
               txt_agg=None,
               txt_pro=None,
               txt_wgh=None,
               txt_ratio=1.0,
               vid_inp=None,
               vid_cont=None,
               vid_wgh=None,
               vid_ratio=1.0,
               pos_enc=None,
               out_tok=None,
               use_mask='nomask',
               same_dim=512,
               vid_bert_params=None,
               txt_bert_params=None,
               agg_dims=None,
               normalize_experts=True):
    super().__init__()

    print('Debug ---> create EMMT.')

    self.sanity_checks = False
    modalities = list(expert_dims.keys())
    self.expert_dims = expert_dims
    self.modalities = modalities
    logger.debug(self.modalities)
    self.mimic_ce_dims = mimic_ce_dims
    self.concat_experts = concat_experts
    self.concat_mix_experts = concat_mix_experts
    self.test_caption_mode = test_caption_mode
    self.freeze_weights = freeze_weights
    self.use_experts = use_experts
    self.use_mask = use_mask
    self.keep_missing_modalities = keep_missing_modalities
    self.l2renorm = l2renorm
    self.same_dim = same_dim
    self.txt_inp = txt_inp
    self.txt_agg = txt_agg
    self.txt_pro = txt_pro
    self.txt_wgh = txt_wgh
    self.vid_inp = vid_inp
    self.vid_cont = vid_cont
    self.vid_wgh = vid_wgh
    self.pos_enc = pos_enc
    self.out_tok = out_tok
    self.vid_bert_params = vid_bert_params
    self.normalize_experts = normalize_experts

    self.rep_dim = rep_dim
    print('Debug ---> set dimension of representations to {}.'.format(rep_dim))
    self.proj_dim = proj_dim
    print('Debug ---> set dimension of projection to {}.'.format(proj_dim))
    self.txt_ratio = txt_ratio
    print('Debug ---> set ratio of representative text tokens to {}.'.format(txt_ratio))
    self.vid_ratio = vid_ratio
    print('Debug ---> set ratio of representative video tokens to {}.'.format(vid_ratio))
    self.embd_out = embd_out
    print('Debug ---> use {} as bert pooling manner.'.format(embd_out))
    self.txt_layers = sorted([int(x) for x in txt_layers.split(',')], reverse=True)
    txt_layers = ','.join(list(map(str, self.txt_layers)))
    print('Debug ---> select layer-{} of text encoder for contrastive learning.'.format(txt_layers))
    self.vid_layers = sorted([int(x) for x in vid_layers.split(',')], reverse=True)
    vid_layers = ','.join(list(map(str, self.vid_layers)))
    print('Debug ---> select layer-{} of video encoder for contrastive learning.'.format(vid_layers))
    self.out_space = out_space
    print('Debug ---> select {} space for retrieval.'.format(out_space))

    # -------------------------------------------------------------------------- video input
    self.video_dim_reduce = nn.ModuleDict()
    for mod in self.modalities:
      in_dim = expert_dims[mod]['dim']
      if self.vid_inp in ['agg', 'both', 'all', 'temp']:
        self.video_dim_reduce[mod] = ReduceDim(in_dim, same_dim)
    # -------------------------------------------------------------------------- video encoder
    # Only Bert architecture is employed for video
    if self.vid_cont == 'bert':
      vid_bert_config = types.SimpleNamespace(**vid_bert_params)
      self.vid_bert = BertModel(vid_bert_config)
    # -------------------------------------------------------------------------- text encoder
    # Only Bert architecture is employed for text or caption
    if self.txt_agg[:4] in ['bert']:
      z = re.match(r'bert([a-z]{3})(\d*)(\D*)', txt_agg)
      assert z
      state = z.groups()[0]
      freeze_until = z.groups()[1]

      # Post aggregation: Use [CLS] token ("cls") or aggregate all tokens
      # (mxp, mnp)
      if z.groups()[2] and z.groups()[2] != 'cls':
        self.post_agg = z.groups()[2]
      else:
        self.post_agg = 'cls'

      if state in ['ftn', 'frz']:
        # State is finetune or frozen, we use a pretrained bert model
        txt_bert_config = 'bert-base-cased'

        # Overwrite config
        if txt_bert_params is None:
          dout_prob = vid_bert_params['hidden_dropout_prob']
          txt_bert_params = {
              'hidden_dropout_prob': dout_prob,
              'attention_probs_dropout_prob': dout_prob,
          }
        self.txt_bert = myTxtBertModel.from_pretrained(txt_bert_config,
                                                     **txt_bert_params)

        if state == 'frz':
          if freeze_until:
            # Freeze only certain layers
            freeze_until = int(freeze_until)
            logger.debug('Freezing text bert until layer %d excluded',
                         freeze_until)
            # Freeze net until given layer
            for name, param in self.txt_bert.named_parameters():
              module = name.split('.')[0]
              if name.split('.')[2].isdigit():
                layer_nb = int(name.split('.')[2])
              else:
                continue
              if module == 'encoder' and layer_nb in range(freeze_until):
                param.requires_grad = False
                logger.debug(name)
          else:
            # Freeze the whole model
            for name, param in self.txt_bert.named_parameters():
              module = name.split('.')[0]
              if module == 'encoder':
                param.requires_grad = False
        else:
          assert not freeze_until

      if self.txt_inp == 'bertfrz':
        # Freeze model
        for param in self.txt_bert.embeddings.parameters():
          param.requires_grad = False
      elif self.txt_inp not in ['bertftn']:
        logger.error('Wrong parameter for the text encoder')
      text_dim = self.txt_bert.config.hidden_size
    else:
      raise "Only Bert architecture can be employed for text"
    # ---------------------------------------------------------------- multi-modal contrastive learning (semantic and feature)
    # transform tokens to representations
    n_txt_layers = len(self.txt_layers)
    n_vid_layers = len(self.vid_layers)
    self.n_layers = n_txt_layers
    assert n_txt_layers == n_vid_layers, 'number of text layers and video layers used to compute contrastive loss must be the same.'
    self.txt_token2rep = nn.ModuleList([ReduceDim(text_dim, self.rep_dim) for _ in range(n_txt_layers)])
    self.vid_token2rep = nn.ModuleList([ReduceDim(same_dim, self.rep_dim) for _ in range(n_vid_layers)])
    # project text and video representations into common space
    print('Debug ---> create model with {} projection head.'.format(projector_layer))
    self.projector_layer = projector_layer
    # sw: shared-weight
    # linear
    # none
    if projector_layer == 'swlinear':
      self.projector_txt = nn.ModuleList([Projector([self.rep_dim, self.proj_dim]) for _ in range(n_txt_layers)])
      self.projector_vid = self.projector_txt
    elif projector_layer == 'nswlinear':
      self.projector_txt = nn.ModuleList([Projector([self.rep_dim, self.proj_dim]) for _ in range(n_txt_layers)])
      self.projector_vid = nn.ModuleList([Projector([self.rep_dim, self.proj_dim]) for _ in range(n_txt_layers)])
    elif projector_layer == 'swnonlinear':
      self.projector_txt = nn.ModuleList([Projector([self.rep_dim, self.rep_dim, self.proj_dim]) for _ in range(n_txt_layers)])
      self.projector_vid = self.projector_txt
    elif projector_layer == 'nswnonlinear':
      self.projector_txt = nn.ModuleList([Projector([self.rep_dim, self.rep_dim, self.proj_dim]) for _ in range(n_txt_layers)])
      self.projector_vid = nn.ModuleList([Projector([self.rep_dim, self.rep_dim, self.proj_dim]) for _ in range(n_txt_layers)])
    elif projector_layer == 'none':
      self.projector_txt = nn.ModuleList([nn.Identity() for _ in range(n_txt_layers)])
      self.projector_vid = nn.ModuleList([nn.Identity() for _ in range(n_txt_layers)])
    # generate each token's weight for subsequent selecting representative ones
    self.txt_w_gen = nn.ModuleList([GenWeight(self.rep_dim, 4) for _ in range(n_txt_layers)])
    self.vid_w_gen = nn.ModuleList([GenWeight(self.rep_dim, 4) for _ in range(n_txt_layers)])

    # added adaptive token ratio
    self.text_adp_ratio = nn.Sequential(
                  nn.Linear(self.rep_dim, self.rep_dim),
                  nn.ReLU(),
                  nn.Linear(self.rep_dim, 1),
                  nn.Sigmoid()
    )

    self.video_adp_ratio = nn.Sequential(
                  nn.Linear(self.rep_dim, self.rep_dim),
                  nn.ReLU(),
                  nn.Linear(self.rep_dim, 1),
                  nn.Sigmoid()
    )

    self.txt_adp_weight = nn.Sequential(
                  TransformerEncoderLayer(),
                  nn.Sigmoid()
    )

    self.vid_adp_weight = nn.Sequential(
                  TransformerEncoderLayer(),
                  nn.Sigmoid()
    )


    # Weights initialization
    #self.apply(self._init_weights)

  def _init_weights(self, module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def select_representative_tokens(self, embd, mod_type, level):
    assert mod_type in ['txt', 'vid'], 'modality type must be text or video'
    # embd_avg = th.mean(embd,dim=1)
    if mod_type == 'txt':
      ratios = self.txt_adp_weight(embd)
      # ratio = self.txt_ratio
      weights = self.txt_w_gen[level](embd)
    elif mod_type == 'vid':
      ratios = self.vid_adp_weight(embd)
      # ratio = self.vid_ratio
      weights = self.vid_w_gen[level](embd)
    # print('weights shape: ', weights.shape)    # [bz, n_tokens]
    selected_tokens = []
    # print('ratio shape:', ratios.shape)        # [bz, 1]
    weighted_token_embd = th.mul(embd, ratios) 
    # logger.info(f'level: {level} \t type: {mod_type} \t ratio: {ratios}')
    # mask = []
    # n_token_selected= th.zeros(embd.shape[0])
    # for idx, ratio in enumerate(ratios):
    #   mask_sample = th.zeros_like(embd[idx])  # weights[idx]   [n_tokens]

    #   n_tokens = int(embd.shape[1] * ratio)  # int 
    #   n_token_selected[idx] = n_tokens
    #   topk_indices = th.topk(weights[idx], k=n_tokens, largest=True)[1] # idx [n_tokens]
    #   topk_indices = th.sort(topk_indices, dim=-1)[0]   # [n_tokens]

    #   mask_sample[topk_indices,:] = 1   
    #   mask.append(mask_sample)

    # mask = th.stack(mask)
    # selected_tokens = th.masked_fill(embd,mask==1, 0)

    # n_tokens = int(embd.shape[1] * ratio)
    # topk_indices = th.topk(weights, k=n_tokens, largest=True)[1]
    # topk_indices = th.sort(topk_indices, dim=1)[0]
    # selected_tokens = [embd[i, topk_indices[i], :] for i in range(embd.shape[0])]
    # selected_tokens = th.stack(selected_tokens)

    return weighted_token_embd

  def forward(self,
              token_ids,
              features,
              features_t,
              features_ind,
              features_avgpool,
              features_maxpool,
              query_masks,
              out='conf',
              device=None,
              debug=None):

    self.device = device
    experts_feats = features
    experts_feats_t = features_t
    experts_feats_ind = features_ind
    ind = {}
    for mod in self.modalities:
      ind[mod] = th.max(experts_feats_ind[mod], 1)[0]
    pooled_experts = {}

    for _, mod in enumerate(self.modalities):
      pooled_experts[f'{mod}_avgpool'] = features_avgpool[mod]
      pooled_experts[f'{mod}_maxpool'] = features_maxpool[mod]

    # Notation: B = batch size, M = number of modalities

    # Unroll repeated captions into present minibatch
    b, captions_per_video, max_text_words, feat_dim = token_ids.size()
    m = len(self.modalities)

    # If Bert architecture is employed
    # --------------------------------------------------------------------------- text encoder *  -----------------------
    if self.txt_agg[:4] in ['bert']:
      token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                 feat_dim)

      input_ids_list = []
      token_type_ids_list = []  # Modality id
      position_ids_list = []  # Position
      attention_mask_list = []  # Valid token or not

      ids_size = (b * captions_per_video,)

      for pos_id in range(max_text_words):
        input_ids_list.append(token_ids[:, pos_id, 0].to(dtype=th.long))
        token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
        position_ids_list.append(th.full(ids_size, pos_id, dtype=th.long))
        attention_mask_list.append(token_ids[:, pos_id, 1].to(dtype=th.long))

      input_ids = th.stack(input_ids_list, dim=1).to(device)
      token_type_ids = th.stack(token_type_ids_list, dim=1).to(device)
      position_ids = th.stack(position_ids_list, dim=1).to(device)
      attention_mask = th.stack(attention_mask_list, dim=1).to(device)

      txt_bert_output = self.txt_bert(input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      head_mask=None)
      #last_layer = txt_bert_output[0]  # [b, max_len, 768]
      # all_layers = [last_layer, last_layer_pooler, all_layers] 
      all_layers = txt_bert_output[2]  # txt_bert_output[1] 是 last layer的pooler

      # compute representations from bert output
      text_reps = [self.txt_token2rep[i](all_layers[layer_id]) for i, layer_id in enumerate(self.txt_layers)]
      text_cls_rep = [layer[:, 0] for layer in text_reps]
      text_token_rep = [layer[:, 1:] for layer in text_reps]

    # ------------------------------------------------------------------------------- video input（不用管） --------------------
    if self.vid_inp in ['agg', 'both', 'all']:
      agg_experts = collections.OrderedDict()
      mnp_experts = collections.OrderedDict()
      maxp_experts = collections.OrderedDict()

      # Embed all features to a common dimension
      for mod in self.modalities:
        layer = self.video_dim_reduce[mod]
        mnp_experts[mod] = layer(pooled_experts[f'{mod}_avgpool'])
        maxp_experts[mod] = layer(pooled_experts[f'{mod}_maxpool'])

      for mod in self.modalities:
        agg_experts[mod] = maxp_experts[mod]

    if self.vid_inp in ['both', 'temp', 'all']:
      for mod in self.modalities:
        layer = self.video_dim_reduce[mod]
        experts_feats[mod] = layer(experts_feats[mod])

    # If Bert architecture is employed                                                  
    # ---------------------------------------------------------------------------- video encoder * -----------------------
    if self.vid_cont == 'bert':
      # 0=[CLS] 1=[SEP] 2=[AGG] 3=[MAXP] 4=[MNP] 5=[VLAD] 6=[FEA]
      input_ids_list = []
      token_type_ids_list = []  # Modality id
      # Position (0 = no position, 1 = unknown, >1 = valid position)
      position_ids_list = []
      features_list = []  # Semantics
      attention_mask_list = []  # Valid token or not

      modality_to_tok_map = collections.OrderedDict()

      # [CLS] token
      tok_id = 0
      ids_size = (b,)
      input_ids_list.append(th.full(ids_size, 0, dtype=th.long))
      token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
      position_ids_list.append(th.full(ids_size, 0, dtype=th.long).to(device))
      features_list.append(
          th.full((b, self.same_dim), 0, dtype=th.float).to(device))
      attention_mask_list.append(th.full(ids_size, 1, dtype=th.long).to(device))

      # Number of temporal tokens per modality
      if self.vid_inp in ['temp', 'both', 'all']:
        max_expert_tokens = collections.OrderedDict()
        for _, modality in enumerate(self.modalities):
          max_expert_tokens[modality] = experts_feats[modality].size()[1]

      # Make the features_t and raw_captions_t start at the minimum value
      if self.pos_enc == 'tint':

        # Clamp the position encoding to [0, max_position_embedding - 1]
        max_pos = self.vid_bert_params['max_position_embeddings'] - 1
        for _, modality in enumerate(self.modalities):
          experts_feats_t[modality].clamp_(min=0, max=max_pos)
          experts_feats_t[modality] = experts_feats_t[modality].long().to(
              device)

      for _, modality in enumerate(self.modalities):
        token_type = self.expert_dims[modality]['idx']

        # Add an aggregated feature token
        if self.vid_inp in ['agg', 'both', 'all']:
          tok_id += 1
          modality_to_tok_map[modality] = tok_id
          input_ids_list.append(th.full(ids_size, 2, dtype=th.long))
          token_type_ids_list.append(
              th.full(ids_size, token_type, dtype=th.long))
          position_ids_list.append(
              th.full(ids_size, 0, dtype=th.long).to(device))
          if self.out_tok == 'sep':
            features_list.append(
                th.full((b, self.same_dim), 0, dtype=th.float).to(device))
          elif self.out_tok == 'mxp':
            features_list.append(maxp_experts[modality])
          elif self.out_tok == 'mnp':
            features_list.append(mnp_experts[modality])
          attention_mask_list.append(ind[modality].to(dtype=th.long).to(device))
        if self.vid_inp in ['temp', 'both', 'all']:
          for frame_id in range(max_expert_tokens[modality]):
            if self.pos_enc == 'ordr':
              position_ids_list.append(
                  th.full(ids_size, frame_id + 1, dtype=th.long).to(device))
            elif self.pos_enc == 'tint':
              position_ids_list.append(experts_feats_t[modality][:, frame_id])
            elif self.pos_enc == 'type':
              position_ids_list.append(
                  th.full(ids_size, 1, dtype=th.long).to(device))
            tok_id += 1
            input_ids_list.append(th.full(ids_size, 6, dtype=th.long))
            token_type_ids_list.append(
                th.full(ids_size, token_type, dtype=th.long))
            features_list.append(experts_feats[modality][:, frame_id, :])
            attention_mask_list.append(
                experts_feats_ind[modality][:, frame_id].to(dtype=th.long))

      features = th.stack(features_list, dim=1).to(self.device)
      input_ids = th.stack(input_ids_list, dim=1).to(self.device)
      token_type_ids = th.stack(token_type_ids_list, dim=1).to(self.device)
      if self.pos_enc != 'none':
        position_ids = th.stack(position_ids_list, dim=1).to(self.device)
      else:
        position_ids = None
      attention_mask = th.stack(attention_mask_list, dim=1).to(self.device)

      # [CLS] {[AGG1][FEA1_1]...[FEA1_30]} {[AGG2][FEA2_1]...[FEA2_30]} ... {[AGG7][FEA7_1]...[FEA7_30]}
      vid_bert_output = self.vid_bert(input_ids, 
                                      attention_mask=attention_mask,  # []
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      features=features)

      #last_layer = vid_bert_output[0]  # [b, max_len, same_dim]
      all_layers = vid_bert_output[2]  # txt_bert_output[1] 是 last layer的pooler

      # compute representations from bert output
      vid_reps = [self.vid_token2rep[i](all_layers[layer_id]) for i, layer_id in enumerate(self.vid_layers)]
      vid_cls_rep = [layer[:, 0] for layer in vid_reps]
      vid_token_rep = [layer[:, 1:] for layer in vid_reps]

    # ------------------------------------------------------------------------------- semantic and feature projection -------------
    # ------------------------------------------------------- tokens筛选模块 * ---------------------------------------------
    # select representative text tokens
    # print('len: ', len(text_token_rep))
    # for idx, text_token_rep_i in enumerate(text_token_rep):
    #   print(f'text shape layer_{idx}:', text_token_rep_i.shape)
    #   print(f'video shape layer_{idx}:', vid_token_rep[idx].shape)
    if self.txt_ratio == 1:
      text_selected_token_rep = text_token_rep
    else:
      text_selected_token_rep = [self.select_representative_tokens(text_token_rep[i], 'txt', i) for i in range(self.n_layers)]
    # select representative video tokens
    if self.vid_ratio == 1:
      vid_selected_token_rep = vid_token_rep
    else:
      vid_selected_token_rep = [self.select_representative_tokens(vid_token_rep[i], 'vid', i) for i in range(self.n_layers)]
    # do average pooling on all tokens
    text_avg_pool_rep = [th.mean(text_token_rep[i], dim=1) for i in range(self.n_layers)]
    vid_avg_pool_rep = [th.mean(vid_token_rep[i], dim=1) for i in range(self.n_layers)]
    # project avg-pooled tokens ([b, rep_dim]) into common space ([b, proj_dim])
    if self.embd_out == 'avg':
      text_embd = [self.projector_txt[i](text_avg_pool_rep[i]) for i in range(self.n_layers)]
      vid_embd = [self.projector_vid[i](vid_avg_pool_rep[i]) for i in range(self.n_layers)]
    else:
      text_embd = [self.projector_txt[i](text_cls_rep[i]) for i in range(self.n_layers)]
      vid_embd = [self.projector_vid[i](vid_cls_rep[i]) for i in range(self.n_layers)]
    # coarse-grained-interaction: compute cross view confusion matrix for contrastive loss * ------------------------------------
    cross_view_conf_matrix = [sharded_cross_view_inner_product(vid_rep=vid_embd[i], txt_rep=text_embd[i]) for i in range(self.n_layers)]
    # project selected representative tokens ([b, n_s_token, rep_dim]) into common space ([b, n_s_token, proj_dim])

    vid_selected_embds = [self.projector_txt[i](vid_selected_token_rep[i]) for i in range(self.n_layers)]
    text_selected_embds = [self.projector_vid[i](text_selected_token_rep[i]) for i in range(self.n_layers)]
    # fine-grained-interaction: compute mean-maximum similarity for self-distillation loss * ------------------------------------
    late_interaction_matrix = [cpt_mean_maximum_similarity(vid_selected_embds[i], text_selected_embds[i]) for i in range(self.n_layers)]
    
    """
    vid_selected_embds = [vid_selected_token_rep[i] for i in range(self.n_layers)]
    text_selected_embds = [text_selected_token_rep[i] for i in range(self.n_layers)]
    # fine-grained-interaction: compute mean-maximum similarity for self-distillation loss
    late_interaction_matrix = [cpt_mean_maximum_similarity(vid_selected_embds[i], text_selected_embds[i]) for i in range(self.n_layers)]
    """

    #late_interaction_matrix = [None for _ in range(self.n_layers)]



    # network output
    if self.out_space == 'rep':
      vid_output_embd = vid_avg_pool_rep[0]
      txt_output_embd = text_avg_pool_rep[0]
    elif self.out_space == 'proj':
      vid_output_embd = vid_embd[0]
      txt_output_embd = text_embd[0]

    return {
      'vid_reps': vid_output_embd,
      'text_reps': txt_output_embd,
      'cross_view_conf_matrix': cross_view_conf_matrix,
      'late_interaction_matrix': late_interaction_matrix
    }
    

class ReduceDim(nn.Module):

  def __init__(self, input_dimension, output_dimension):
    super(ReduceDim, self).__init__()
    self.fc = nn.Linear(input_dimension, output_dimension)

  def forward(self, x):
    x = self.fc(x)
    x = F.normalize(x, dim=-1)
    return x


class GenWeight(nn.Module):
  def __init__(self, in_dim, reduction_scale):
      super().__init__()
      intermediate_dim = in_dim // reduction_scale
      self.w_G = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_dim, intermediate_dim),
        nn.ReLU(inplace=True),
        nn.Linear(intermediate_dim, 1)
      )

  def forward(self, tokens):
    bs = tokens.shape[0]
    tokens_w = F.softmax(self.w_G(tokens).view(bs, -1), dim=1)
    return tokens_w


def sharded_cross_view_inner_product(vid_rep,
                                     txt_rep,
                                     subspaces=None):
  """Compute similarities between all captions and videos."""
  b = vid_rep.shape[0]
  num_caps = txt_rep.shape[0] // b

  txt_rep_norm = F.normalize(txt_rep, dim=1)
  vid_rep_norm = F.normalize(vid_rep, dim=1)
  sims = txt_rep_norm @ vid_rep_norm.T  # th.matmul

  if num_caps > 1:
    # aggregate similarities from different captions
    sims = sims.view(b, num_caps, b)
    sims = th.mean(sims, dim=1)
    sims = sims.view(b, b)
  return sims

def cpt_mean_maximum_similarity(vid_rep,
                                txt_rep):
  # vid_rep: [b, 218, 512]
  # txt_rep: [b, 30, 512]
  # txt_n_tokens = txt_n_tokens.unsqueeze(-1)
  # vid_n_tokens = vid_n_tokens.unsqueeze(0)
  # print('txt_token shape:', txt_n_tokens.shape)
  # print('vid_token shape:', vid_n_tokens.shape)
  txt_bs, txt_n_tokens, txt_dim = txt_rep.shape
  vid_bs, vid_n_tokens, vid_dim = vid_rep.shape
  txt_rep_norm = F.normalize(txt_rep, dim=-1)
  vid_rep_norm = F.normalize(vid_rep, dim=-1)
  txt_rep_norm = txt_rep_norm.view(txt_bs * txt_n_tokens, txt_dim)  # [bs*218, 512]
  vid_rep_norm = vid_rep_norm.view(vid_bs * vid_n_tokens, vid_dim)  # [bs*30, 512]
  sim = th.matmul(txt_rep_norm, vid_rep_norm.T)  # [bs*218, bs*30]
  sim = sim.view(txt_bs, txt_n_tokens, vid_bs, vid_n_tokens)  # [bs, 30, bs, 218]
  t2v = th.mean(th.max(sim, dim=-1)[0], dim=1)  # [bs, bs]
  v2t = th.mean(th.max(sim, dim=1)[0], dim=-1)  # [bs, bs]
  # print('t2v:', th.max(sim, dim=-1)[0].shape)
  # print('v2t:', th.max(sim, dim=1)[0].shape)
  # t2v = th.sum(th.max(sim, dim=-1)[0], dim=1)/ txt_n_tokens # [bs, bs]
  # v2t = th.mean(th.max(sim, dim=1)[0], dim=-1)/ vid_n_tokens  # [bs, bs]

  return {
    "t2v": t2v,
    "v2t": v2t
  }

def cpt_cross_corrletaion_matrix(vid_embd_CL,
                                 txt_embd_CL,
                                 subspaces):
  """Compute cross-correlation matrix."""

  #vid_embd_CL_norm = F.normalize(vid_embd_CL, dim=1)
  #txt_embd_CL_norm = F.normalize(txt_embd_CL, dim=1)
  b, d = vid_embd_CL.shape
  num_caps = txt_embd_CL.shape[0] // b

  #txt_rep_norm = F.normalize(txt_embd_CL, dim=0)
  #vid_rep_norm = F.normalize(vid_embd_CL, dim=0)

  if num_caps == 1: 
    cross_corr_mat = txt_embd_CL.T @ vid_embd_CL
  elif num_caps > 1:
    txt_embd_CL = txt_embd_CL.view(b, num_caps, -1)
    cross_corr_mat = th.zeros(d, d).to(vid_embd_CL.device)
    for i in range(num_caps):
      cross_corr_mat += (txt_embd_CL[:, i, :].T @ vid_embd_CL)
    cross_corr_mat /= num_caps

  return cross_corr_mat

class Projector(nn.Module):
  def __init__(self, sizes):
    super().__init__()
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        #layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    self.projector = nn.Sequential(*layers)

  def forward(self, token_embds):
    embd_cl = self.projector(token_embds)
    return embd_cl


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=512, num_attention_heads=4, intermediate_size=2048):
        super(TransformerEncoderLayer, self).__init__()

        # 构建 BertConfig 对象
        config = BertConfig(hidden_size=hidden_size,
                            num_attention_heads=num_attention_heads,
                            intermediate_size=intermediate_size)

        # 构建自注意力层、中间层和输出层
        self.attention = BertSelfAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 自注意力层
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions)
        
        # print('type attn out:', type(attention_output))
        # print('len attn out', len(attention_output))
        # print('shape attn out', attention_output[0].shape)
        # 中间层
        intermediate_output = self.intermediate(attention_output[0])
        # 输出层
        encoder_output = self.output(intermediate_output, attention_output[0])

        # for sigmoid:
        output = self.proj(encoder_output)*1000
        if output_attentions:
            return output, attention_output
        else:
            return output



if __name__ == '__main__':
  txt_rep = F.normalize(th.randn(2, 2, 3), dim=1).cuda(1)
  vid_rep = F.normalize(th.randn(2, 2, 3), dim=1).cuda(1)
