# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())

    def compute_logits(self, nl_vecs, code_vecs, bs):
        if len(code_vecs) < bs:
            code_vecs = np.pad(code_vecs, ((0, bs-len(code_vecs)), (0, 0)), 'constant', constant_values=(0, 0))
        if len(nl_vecs) < bs:
            nl_vecs = np.pad(nl_vecs, ((0, bs-len(nl_vecs)), (0, 0)), 'constant', constant_values=(0, 0))
        code_vecs = torch.tensor(code_vecs).cuda()
        nl_vecs = torch.tensor(nl_vecs).cuda()
        nl_vec2 = nl_vecs.unsqueeze(1).repeat([1, bs, 1])
        code_vec2 = code_vecs.unsqueeze(0).repeat([bs, 1, 1])
        logits = self.mlp(torch.cat((nl_vec2, code_vec2, nl_vec2-code_vec2, nl_vec2*code_vec2), 2)).squeeze(2)
        return logits

    def compute_loss(self, logits, mask):
        return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_negative_mask(self, batch_size):
        mask = torch.diag(torch.ones(batch_size)).float()
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def _get_positive_mask(self, batch_size):
        mask = torch.diag(torch.ones(batch_size)).float()
        return mask.cuda(non_blocking=True)

    def forward(self, code_inputs, nl_inputs, code_aug_inputs, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs, code_aug_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:bs*2]
        code_aug_vec = outputs[bs*2:]

        if return_vec:
            return code_vec, nl_vec

        scores = (nl_vec[:, None, :]*code_vec[None, :, :]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss_ce = loss_fct(scores, torch.arange(bs, device=scores.device))

        nl_vec2 = nl_vec.unsqueeze(1).repeat([1, bs, 1])
        code_vec2 = code_vec.unsqueeze(0).repeat([bs, 1, 1])
        logits = self.mlp(torch.cat((nl_vec2, code_vec2, nl_vec2-code_vec2, nl_vec2*code_vec2), 2)).squeeze(2)

        matrix_labels = torch.diag(torch.ones(bs)).float()  # (Batch, Batch)
        poss = logits[matrix_labels == 1]
        negs = logits[matrix_labels == 0]
        loss_cls = - (torch.log(1 - negs + 1e-5).mean() + torch.log(poss + 1e-5).mean())

        code_features = nn.functional.normalize(code_vec, dim=1)
        nl_features = nn.functional.normalize(nl_vec, dim=1)
        code_aug_features = nn.functional.normalize(code_aug_vec, dim=1)

        logits_code_p = code_features @ nl_features.t()
        logits_nl_p = nl_features @ code_features.t()
        logits_code_n = code_features @ code_features.t()
        logits_nl_n = nl_features @ nl_features.t()
        logits_code_p_aug = code_features @ code_aug_features.t()
        logits_nl_p_aug = nl_features @ code_aug_features.t()

        temperature = 0.03
        logits_code_p /= temperature
        logits_nl_p /= temperature
        logits_code_n /= temperature
        logits_nl_n /= temperature
        logits_code_p_aug /= temperature
        logits_nl_p_aug /= temperature

        negative_mask = self._get_negative_mask(bs)
        logits_code_n = logits_code_n * negative_mask
        logits_nl_n = logits_nl_n * negative_mask

        code_logits = torch.cat([logits_code_p, 0.8 * logits_code_n, 0.1 * logits_code_p_aug], dim=1)
        nl_logits = torch.cat([logits_nl_p, 0.8 * logits_nl_n, 0.1 * logits_nl_p_aug], dim=1)
        mask_code_p = torch.diag(torch.ones(bs)).cuda()
        mask_nl_p = torch.diag(torch.ones(bs)).cuda()
        mask_code_n = torch.zeros_like(logits_code_n)
        mask_nl_n = torch.zeros_like(logits_nl_n)
        mask_code_p_aug = torch.diag(torch.ones(bs)).cuda()
        mask_nl_p_aug = torch.diag(torch.ones(bs)).cuda()
        mask_code = torch.cat([mask_code_p, mask_code_n, mask_code_p_aug], dim=1)
        mask_nl = torch.cat([mask_nl_p, mask_nl_n, mask_nl_p_aug], dim=1)

        loss_code = self.compute_loss(code_logits, mask_code)
        loss_nl = self.compute_loss(nl_logits, mask_nl)
        loss_clr = (loss_code.mean() + loss_nl.mean()) / 2

        loss = loss_ce + 0.1 * loss_clr + loss_cls

        return loss, code_vec, nl_vec
