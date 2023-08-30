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
"""Training losses.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""
import torch as th
import torch.nn as nn
import numpy as np
try:
  from model.HiEMMT import sharded_cross_view_inner_product
except:
  pass
import torch.nn.functional as F


class MaxMarginRankingLoss(nn.Module):
  """Implementation of the Max-margin ranking loss."""

  def __init__(self, margin=1, fix_norm=True):
    super().__init__()
    self.fix_norm = fix_norm
    self.loss = th.nn.MarginRankingLoss(margin)
    self.margin = margin

  def forward(self, x):
    n = x.size()[0]

    x1 = th.diag(x)
    x1 = x1.unsqueeze(1)
    x1 = x1.expand(n, n)
    x1 = x1.contiguous().view(-1, 1)
    x1 = th.cat((x1, x1), 0)

    x2 = x.view(-1, 1)
    x3 = x.transpose(0, 1).contiguous().view(-1, 1)

    x2 = th.cat((x2, x3), 0)
    max_margin = F.relu(self.margin - (x1 - x2))

    if self.fix_norm:
      # remove the elements from the diagonal
      keep = th.ones(x.shape) - th.eye(x.shape[0])
      keep1 = keep.view(-1, 1)
      keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
      keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
      if x1.is_cuda:
        keep_idx = keep_idx.cuda()
      x1_ = th.index_select(x1, dim=0, index=keep_idx)
      x2_ = th.index_select(x2, dim=0, index=keep_idx)
      max_margin = F.relu(self.margin - (x1_ - x2_))

    return max_margin.mean()


class InfoNceLoss(nn.Module):
  """Implementation of the noise-constrastive estimation loss."""
  """
  def __init__(self):
    super().__init__()
    self.loss = th.nn.CrossEntropyLoss(reduction='mean')

  def forward(self, x):
    n = x.size()[0]
    target = th.arange(n)
    if x.is_cuda:
      target = target.cuda()

    return self.loss(x, target) + self.loss(th.transpose(x, 0, 1), target)
  """
  def __init__(self, temperature=0.07):
    super().__init__()
    self.loss = th.nn.CrossEntropyLoss()
    self.T = temperature

  def forward(self, txt_reps, vid_reps):
    x = sharded_cross_view_inner_product(
        txt_rep=txt_reps,
        vid_rep=vid_reps
    )
    x /= self.T
    n = x.size()[0]
    target = th.arange(n)
    if x.is_cuda:
      target = target.cuda()

    return self.loss(x, target) + self.loss(th.transpose(x, 0, 1), target)

class BarlowTwinsLoss(nn.Module):
  """Implementation of the Barlow Twins loss."""

  def __init__(self, lambd=0.0051):
    super().__init__()
    self.lambd =lambd
    self.loss = nn.CrossEntropyLoss()
  
  def off_diagonal(self, x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

  def forward(self, c):
    on_diag = th.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = self.off_diagonal(c).pow_(2).sum()
    loss = on_diag + self.lambd * off_diag
    return loss


class MixLoss(nn.Module):
  def __init__(self, temperature=0.07, lambd=0.00, start_distill_epoch=10):
    super().__init__()
    self.ce_loss = th.nn.CrossEntropyLoss()
    self.kl_loss = th.nn.functional.kl_div
    self.T = temperature
    self.lambd = lambd
    self.start_distill_epoch = start_distill_epoch

  def forward_0(self, global_conf_mat, token_wise_conf_mat):
    global_conf_mat /= self.T
    n = global_conf_mat.size()[0]
    target = th.arange(n).to(global_conf_mat.device)

    t2v_conf_mat = global_conf_mat
    v2t_conf_mat = th.transpose(global_conf_mat, 0, 1)

    t2v_soft_target = F.softmax(token_wise_conf_mat["t2v"], dim=-1)
    v2t_soft_target = F.softmax(token_wise_conf_mat["v2t"], dim=-1)

    global_loss = self.ce_loss(t2v_conf_mat, target) + self.ce_loss(v2t_conf_mat, target)
    t2v_distill_loss = self.kl_loss(F.log_softmax(t2v_conf_mat, dim=-1), t2v_soft_target)
    v2t_distill_loss = self.kl_loss(F.log_softmax(v2t_conf_mat, dim=-1), v2t_soft_target)
    distill_loss = t2v_distill_loss + v2t_distill_loss
    assert t2v_distill_loss > 0, "negtaive loss of {}".format(t2v_distill_loss)
    assert v2t_distill_loss > 0, "negtaive loss of {}".format(v2t_distill_loss)

    total_loss = global_loss + self.lambd * distill_loss

    return total_loss

  def forward_memory_bank(self, vid_logit_label, txt_logit_label):
    vid_logits, vid_labels = vid_logit_label['logits'], vid_logit_label['labels']
    txt_logits, txt_labels = txt_logit_label['logits'], txt_logit_label['labels']

    vid_logits /= self.T
    txt_logits /= self.T

    global_loss = self.ce_loss(vid_logits, vid_labels) + self.ce_loss(txt_logits, txt_labels)

    """
    n = vid_logits.size()[0]
    t2v_conf_mat = cross_view_conf_matrix
    v2t_conf_mat = th.transpose(cross_view_conf_matrix, 0, 1)

    t2v_soft_target = F.softmax(late_interaction_matrix["t2v"], dim=-1)
    v2t_soft_target = F.softmax(late_interaction_matrix["v2t"], dim=-1)

    t2v_distill_loss = self.kl_loss(F.log_softmax(t2v_conf_mat, dim=-1), t2v_soft_target)
    v2t_distill_loss = self.kl_loss(F.log_softmax(v2t_conf_mat, dim=-1), v2t_soft_target)
    distill_loss = t2v_distill_loss + v2t_distill_loss
    assert t2v_distill_loss > 0, "negtaive loss of {}".format(t2v_distill_loss)
    assert v2t_distill_loss > 0, "negtaive loss of {}".format(v2t_distill_loss)

    total_loss = global_loss + self.lambd * distill_loss 
    """

    return global_loss

  def forward(self, output, is_distill=False):
    vid_reps, txt_reps = output['vid_reps'], output['text_reps']
    bs = vid_reps.size(0)

    # compute InfoNceLoss
    cross_view_conf_matrix = sharded_cross_view_inner_product(vid_rep=vid_reps, txt_rep=txt_reps)
    cross_view_conf_matrix = cross_view_conf_matrix / self.T

    t2v_conf_mat = cross_view_conf_matrix
    v2t_conf_mat = th.transpose(cross_view_conf_matrix, 0, 1)

    target = th.arange(bs).to(vid_reps.device)
    infonce_loss = self.ce_loss(t2v_conf_mat, target) + self.ce_loss(v2t_conf_mat, target)

    # compute self-distillation loss
    if is_distill:
      late_interaction_matrix = output['late_interaction_matrix']

      t2v_soft_target = F.softmax(late_interaction_matrix["t2v"], dim=-1)
      v2t_soft_target = F.softmax(late_interaction_matrix["v2t"], dim=-1)

      t2v_distill_loss = self.kl_loss(F.log_softmax(t2v_conf_mat, dim=-1), t2v_soft_target)
      v2t_distill_loss = self.kl_loss(F.log_softmax(v2t_conf_mat, dim=-1), v2t_soft_target)
      distill_loss = t2v_distill_loss + v2t_distill_loss
      assert t2v_distill_loss > 0, "negtaive loss of {}".format(t2v_distill_loss)
      assert v2t_distill_loss > 0, "negtaive loss of {}".format(v2t_distill_loss)
    
    else:
      distill_loss = 0

    # compute total loss 
    total_loss = infonce_loss + self.lambd * distill_loss 

    return total_loss


class RawMixLoss(nn.Module):
  def __init__(self, temperature=0.07, lambd=0.00, start_distill_epoch=10):
    super().__init__()
    self.ce_loss = th.nn.CrossEntropyLoss()
    self.kl_loss = th.nn.functional.kl_div
    self.T = temperature
    self.lambd = lambd
    self.start_distill_epoch = start_distill_epoch

  def forward(self, global_conf_mat, token_wise_conf_mat, distill=False):
    global_conf_mat /= self.T
    n = global_conf_mat.size()[0]
    target = th.arange(n).to(global_conf_mat.device)

    t2v_conf_mat = global_conf_mat
    v2t_conf_mat = th.transpose(global_conf_mat, 0, 1)

    t2v_soft_target = F.softmax(token_wise_conf_mat["t2v"], dim=-1)
    v2t_soft_target = F.softmax(token_wise_conf_mat["v2t"], dim=-1)

    global_loss = self.ce_loss(t2v_conf_mat, target) + self.ce_loss(v2t_conf_mat, target)
    if not distill:
      return global_loss

    t2v_distill_loss = self.kl_loss(F.log_softmax(t2v_conf_mat, dim=-1), t2v_soft_target)
    v2t_distill_loss = self.kl_loss(F.log_softmax(v2t_conf_mat, dim=-1), v2t_soft_target)
    distill_loss = t2v_distill_loss + v2t_distill_loss
    assert t2v_distill_loss > 0, "negtaive loss of {}".format(t2v_distill_loss)
    assert v2t_distill_loss > 0, "negtaive loss of {}".format(v2t_distill_loss)

    total_loss = global_loss + self.lambd * distill_loss

    return total_loss


class HiMixLoss(nn.Module):
  def __init__(self, temperature=0.07, lambd=0., gamma=0., start_distill_epoch=0, layer_weights=None, HiSD_weights=None):
    super().__init__()
    self.ce_loss = th.nn.CrossEntropyLoss()
    self.kl_loss = nn.KLDivLoss()
    self.T = temperature
    self.lambd = lambd
    self.gamma = gamma
    self.start_distill_epoch = start_distill_epoch
    self.layer_weights = layer_weights
    self.HiSD_weights = HiSD_weights

  def forward(self, cross_view_conf_matrix, late_interaction_matrix, distill=False, w=1):
    # InfoNCE Loss
    cross_view_conf_matrix /= self.T

    n = cross_view_conf_matrix.size()[0]
    target = th.arange(n).to(cross_view_conf_matrix.device)

    t2v_conf_mat = cross_view_conf_matrix 
    v2t_conf_mat = th.transpose(cross_view_conf_matrix, 0, 1)


    global_loss = self.ce_loss(t2v_conf_mat, target) + self.ce_loss(v2t_conf_mat, target)
    if not distill:
      return global_loss

    # cross-grained self-distillation loss
    t2v_soft_target = F.softmax(late_interaction_matrix["t2v"], dim=-1)
    v2t_soft_target = F.softmax(late_interaction_matrix["v2t"], dim=-1)
    t2v_distill_loss = self.kl_loss(F.log_softmax(t2v_conf_mat, dim=-1), t2v_soft_target)
    v2t_distill_loss = self.kl_loss(F.log_softmax(v2t_conf_mat, dim=-1), v2t_soft_target)
    distill_loss = t2v_distill_loss + v2t_distill_loss
    assert t2v_distill_loss > 0, "negtaive loss of {}".format(t2v_distill_loss)
    assert v2t_distill_loss > 0, "negtaive loss of {}".format(v2t_distill_loss)

    total_loss = w * (global_loss + self.lambd * distill_loss)

    return total_loss


class MomentumMixLoss(nn.Module):
  def __init__(self, temperature=0.07, lambd=0.00, start_distill_epoch=10):
    super().__init__()
    self.ce_loss = th.nn.CrossEntropyLoss()
    self.kl_loss = th.nn.functional.kl_div
    self.T = temperature
    self.lambd = lambd
    self.start_distill_epoch = start_distill_epoch

  def forward(self,  model_out, distill=False):
    v2t, t2v = model_out['v2t'], model_out['t2v']
    vid_logits, vid_labels = v2t['logits'], v2t['labels']
    txt_logits, txt_labels = t2v['logits'], t2v['labels']

    vid_logits /= self.T
    txt_logits /= self.T

    global_loss = self.ce_loss(vid_logits, vid_labels) + self.ce_loss(txt_logits, txt_labels)
    if not distill:
      return global_loss

    t2v_conf_mat = model_out['cross_view_conf_matrix']
    v2t_conf_mat = v2t_conf_mat = th.transpose(t2v_conf_mat, 0, 1)

    token_wise_conf_mat = model_out['late_interaction_matrix']
    t2v_soft_target = F.softmax(token_wise_conf_mat["t2v"], dim=-1)
    v2t_soft_target = F.softmax(token_wise_conf_mat["v2t"], dim=-1)

    t2v_distill_loss = self.kl_loss(F.log_softmax(t2v_conf_mat, dim=-1), t2v_soft_target)
    v2t_distill_loss = self.kl_loss(F.log_softmax(v2t_conf_mat, dim=-1), v2t_soft_target)
    distill_loss = t2v_distill_loss + v2t_distill_loss
    assert t2v_distill_loss > 0, "negtaive loss of {}".format(t2v_distill_loss)
    assert v2t_distill_loss > 0, "negtaive loss of {}".format(v2t_distill_loss)

    total_loss = global_loss + self.lambd * distill_loss

    return total_loss

  
class OneWayDCLoss(nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(OneWayDCLoss, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.SMALL_NUM = np.log(1e-45)

    def forward(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        cross_view_distance = th.mm(z1, z2.t())
        positive_loss = -th.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = th.cat((th.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = th.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = th.logsumexp(neg_similarity + neg_mask * self.SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class DCLoss(nn.Module):
  def __init__(self, temperature=0.1, weight_fn=None):
    super(DCLoss, self).__init__()
    self.one_way_loss = OneWayDCLoss(temperature=temperature, weight_fn=weight_fn)
  
  def forward(self, z1, z2):
    loss = self.one_way_loss(z1, z2) + self.one_way_loss(z2, z1)
    return loss


class DCLMixLoss(nn.Module):
  def __init__(self, temperature=0.07, lambd=0, start_distill_epoch=0):
    super().__init__()
    self.one_way_loss = OneWayDCLoss(temperature=temperature)
    self.kl_loss = th.nn.functional.kl_div
    self.T = temperature
    self.lambd = lambd
    self.start_distill_epoch = start_distill_epoch

  def forward(self, vid_rep, txt_rep, token_wise_conf_mat, distill=False):
    global_conf_mat = sharded_cross_view_inner_product(
      vid_rep=vid_rep,
      txt_rep=txt_rep
    )
    global_conf_mat /= self.T

    t2v_conf_mat = global_conf_mat
    v2t_conf_mat = th.transpose(global_conf_mat, 0, 1)

    t2v_soft_target = F.softmax(token_wise_conf_mat["t2v"], dim=-1)
    v2t_soft_target = F.softmax(token_wise_conf_mat["v2t"], dim=-1)

    global_loss = self.one_way_loss(vid_rep, txt_rep) + self.one_way_loss(txt_rep, vid_rep)
    if not distill:
      return global_loss

    t2v_distill_loss = self.kl_loss(F.log_softmax(t2v_conf_mat, dim=-1), t2v_soft_target)
    v2t_distill_loss = self.kl_loss(F.log_softmax(v2t_conf_mat, dim=-1), v2t_soft_target)
    distill_loss = t2v_distill_loss + v2t_distill_loss
    assert t2v_distill_loss > 0, "negtaive loss of {}".format(t2v_distill_loss)
    assert v2t_distill_loss > 0, "negtaive loss of {}".format(v2t_distill_loss)

    total_loss = global_loss + self.lambd * distill_loss

    return total_loss


def sharded_cross_view_inner_product(vid_rep,
                                     txt_rep,
                                     subspaces=None):
  """Compute similarities between all captions and videos."""
  b = vid_rep.shape[0]
  num_caps = txt_rep.shape[0] // b

  txt_rep_norm = F.normalize(txt_rep, dim=-1)
  vid_rep_norm = F.normalize(vid_rep, dim=-1)
  sims = txt_rep_norm @ vid_rep_norm.T

  if num_caps > 1:
    # aggregate similarities from different captions
    sims = sims.view(b, num_caps, b)
    sims = th.mean(sims, dim=1)
    sims = sims.view(b, b)
  return sims


if __name__ == '__main__':
  """  
  infoNCE = InfoNceLoss()
  confusion_matrix = th.randn([32, 32])
  loss = infoNCE(confusion_matrix)
  print(loss.item())

  btloss = BarlowTwinsLoss()
  correlation_matrix = th.randn([512, 512])
  loss = btloss(correlation_matrix)
  print(loss.item())
  """
  """
  dcl_loss = DCLoss(temperature=0.07).cuda()
  z1 = th.randn(4, 16).cuda()
  z2 = th.randn(4, 16).cuda()
  z1 = F.normalize(z1, dim=1)
  z2 = F.normalize(z2, dim=1)
  loss = dcl_loss(z1, z2)
  print(loss.item())
  loss = dcl_loss(z2, z1)
  print(loss.item())
  """
  mix_loss = MixLoss(temperature=0.07, lambd=30)
  z1 = th.randn(4, 16).cuda()
  z2 = th.randn(4, 16).cuda()
  z1 = F.normalize(z1, dim=1)
  z2 = F.normalize(z2, dim=1)
  late_interaction_matrix = {
    't2v': th.randn(4, 4).cuda(),
    'v2t': th.randn(4, 4).cuda()
  }
  output = {
    'vid_reps': z1,
    'text_reps': z2,
    'late_interaction_matrix': late_interaction_matrix
  }
  loss = mix_loss(output)
  print(loss.item())
  