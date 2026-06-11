import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.tal.assigner import TaskAlignedAssigner
from utils.torch_utils import de_parallel

def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps

class VarifocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # 1. Prepare Masks and Selections
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        # 2. Standard CIoU Calculation
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        
        # 3. Focaler-IoU Logic (Paper Implementation for Small Tumors)
        device = iou.device
        d, u = 0.0, 0.95 # Optimal thresholds for medical imaging
        focaler_iou = ((iou - d) / (u - d)).clamp(0, 1)
        focaler_iou = torch.where(iou <= u, focaler_iou, torch.full_like(iou, 1.0))
        
        loss_iou = (1.0 - focaler_iou) * bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # 4. DFL loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

class ComputeLoss:
    # Compute losses main class
    def __init__(self, model, use_dfl=True):
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))
        g = h["fl_gamma"]
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.nl = m.nl
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.assigner2 = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.bbox_loss2 = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max().int(), 5, device=self.device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None, epoch=0):
        loss = torch.zeros(3, device=self.device)
        feats = p[1][0] if isinstance(p, tuple) else p[0]
        feats2 = p[1][1] if isinstance(p, tuple) else p[1]

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        pred_distri2, pred_scores2 = torch.cat([xi.view(feats2[0].shape[0], self.no, -1) for xi in feats2], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores2 = pred_scores2.permute(0, 2, 1).contiguous()
        pred_distri2 = pred_distri2.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes2 = self.bbox_decode(anchor_points, pred_distri2)

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        
        target_labels2, target_bboxes2, target_scores2, fg_mask2 = self.assigner2(
            pred_scores2.detach().sigmoid(), (pred_bboxes2.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)
        target_bboxes2 /= stride_tensor
        target_scores_sum2 = max(target_scores2.sum(), 1)

        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        loss[1] = loss[1] * 0.25 + self.BCEcls(pred_scores2, target_scores2.to(dtype)).sum() / target_scores_sum2

        if fg_mask.sum():
            l_box, l_dfl, _ = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
            loss[0] += l_box * 0.25
            loss[2] += l_dfl * 0.25
        if fg_mask2.sum():
            l_box2, l_dfl2, _ = self.bbox_loss2(pred_distri2, pred_bboxes2, anchor_points, target_bboxes2, target_scores2, target_scores_sum2, fg_mask2)
            loss[0] += l_box2
            loss[2] += l_dfl2

        loss[0] *= 7.5
        loss[1] *= 0.5
        loss[2] *= 1.5

        return loss.sum() * batch_size, loss.detach()