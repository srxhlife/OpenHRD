import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import math
import random
from scipy.optimize import linear_sum_assignment
from randaugment import RandAugmentMC

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class TransformTwice:
    def __init__(self, transform, test=0):
        self.owssl = transform
        self.test = test

        self.simclr = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.fixmatch_weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.fixmatch_strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def __call__(self, inp):
        out1 = self.owssl(inp)
        out2 = self.owssl(inp)
        out3 = self.fixmatch_strong(inp)
        if self.test == 0:
            return out1, out2, out3
        if self.test == 1:
            return out1

class Cifar10TransformTwice:
    def __init__(self, transform, test=0):
        self.owssl = transform
        self.test = test
        self.simclr = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        self.fixmatch_strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def __call__(self, inp):
        out1 = self.owssl(inp)
        out2 = self.owssl(inp)
        out3 = self.fixmatch_strong(inp)
        if self.test == 0:
            return out1, out2, out3
        if self.test == 1:
            return out1

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)
    return res

def cluster_acc(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

class Centroid_Constraint_Class(nn.Module):
    def __init__(self, num_classes, feature_dim, tau_a=0.1 , momentum=0.9):
        super().__init__()
        self.tau_a = tau_a
        self.momentum = momentum
        self.num_classes = num_classes
        self.register_buffer("centroids", torch.zeros(num_classes, feature_dim))
        self.register_buffer("counts", torch.zeros(num_classes))

    def update_centroids(self, z, pseudo_labels):
        with torch.no_grad():
            for k in range(self.num_classes):
                mask = (pseudo_labels == k)
                if mask.sum() == 0:
                    continue
                batch_k_mean = z[mask].mean(dim=0)
                if self.counts[k] == 0:
                    self.centroids[k] = batch_k_mean
                else:
                    self.centroids[k] = (self.momentum * self.centroids[k] + (1 - self.momentum) * batch_k_mean)
                self.counts[k] += mask.sum().item()

    def forward(self, z_x, z_x_hat):
        sim_matrix = F.cosine_similarity(z_x_hat.unsqueeze(1), self.centroids.unsqueeze(0), dim=2)
        centroid_labels = torch.argmax(sim_matrix, dim=1)
        sim_matrix_x = F.cosine_similarity(z_x.unsqueeze(1),self.centroids.unsqueeze(0),dim=2) / self.tau_a
        numerator = torch.exp(sim_matrix_x[torch.arange(z_x.size(0)), centroid_labels])
        denominator = torch.exp(sim_matrix_x).sum(dim=1)
        losses = -torch.log(numerator / denominator) / 2
        return losses.mean()

class WarmupScheduler:
    def __init__(self, switch_epoch=100, mode='binary'):
        self.switch_epoch = switch_epoch
        self.mode = mode

    def get_coeff(self, epoch):
        if self.mode == 'binary':
            return 1.0 if epoch <= self.switch_epoch else 0.0
        elif self.mode == 'linear':
            return max(0.0, 1.0 - epoch / self.switch_epoch)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

def representation_constraint(prob_row, matched_prob_col, prob_w_matched, prob_w, feat2, matched_feats, centroid_constraint, bce, rc_flag, epoch):
    warmup = WarmupScheduler()
    pos_sim = torch.bmm(prob_row, matched_prob_col).squeeze()
    ones = torch.ones_like(pos_sim)
    if rc_flag == 'func':
        num_samples, num_classes = prob_w.shape
        pseudo_class = prob_w.argmax(dim=1)
        unique_labels, counts = torch.unique(pseudo_class, return_counts=True)
        labels_count = torch.zeros(num_classes, dtype=torch.long, device=prob_w.device)
        labels_count[unique_labels] = counts
        mask = torch.zeros(num_samples, num_classes, num_classes, device=prob_w.device)
        mask[torch.arange(num_samples), pseudo_class] = 1
        centroids = torch.einsum("nkd,nd->kd", mask, prob_w)
        centroids = centroids / labels_count.view(-1, 1).float().clamp(min=1e-10)
        sim_matrix_x = torch.mm(prob_w_matched, centroids.t()).div(0.07)
        return bce(pos_sim, ones) + F.cross_entropy(sim_matrix_x, pseudo_class, weight=None) * warmup.get_coeff(epoch) / 2
    elif rc_flag == 'class':
        with torch.no_grad():
            pseudo_labels = torch.argmax(
                F.cosine_similarity(matched_feats.unsqueeze(1), centroid_constraint.centroids.unsqueeze(0), dim=2),dim=1)
        centroid_constraint.update_centroids(matched_feats, pseudo_labels)
        return bce(pos_sim, ones) + centroid_constraint(feat2, matched_feats) * warmup.get_coeff(epoch)
    else:
        raise ValueError(f'rc_flag is invalid')

def debiasing_regularization(current_logit, beta, eta, known_novel_gap, split_class):
    tau_known = eta - known_novel_gap
    tau_novel = eta + known_novel_gap
    log_beta = torch.log(beta)
    log_beta[:,:split_class] *= tau_known
    log_beta[:,split_class:] *= tau_novel
    debiased_prob = F.softmax(current_logit - eta * log_beta, dim=1)
    return debiased_prob

def initialize_beta(class_num=100):
    beta = (torch.ones([1, class_num], dtype=torch.float)/class_num).cuda()
    return beta

def update_beta(probs, beta, momentum, beta_mask=None):
    if beta_mask is not None:
        mean_prob = probs.detach() * beta_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    beta = momentum * beta + (1 - momentum) * mean_prob
    return beta

class NTA_Class(nn.Module):
    def __init__(self, class_num, known_class_num, temperature, device):
        super(NTA_Class, self).__init__()
        self.class_num = class_num
        self.known_class_num = known_class_num
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_clusters(class_num)
        self.ce = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, prob, prob_s):
        P_n = prob.t()
        P_n_s = prob_s.t()
        N = 2 * self.class_num
        P = torch.cat((P_n, P_n_s), dim=0)
        sim = self.similarity_f(P.unsqueeze(1), P.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        pos_sim= torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        neg_sim = sim[self.mask].reshape(N, -1)
        neg_sim[:int(self.known_class_num), : int(self.known_class_num-1)] = 0
        neg_sim[int(self.class_num): int(self.class_num + self.known_class_num), : int(self.known_class_num-1)] = 0
        sim_labels = torch.zeros(N).to(pos_sim.device).long()
        sim_cat = torch.cat((pos_sim, neg_sim), dim=1)
        marginal_prob = prob.sum(0).view(-1)
        marginal_prob /= marginal_prob.sum()
        marginal_prob_s = prob_s.sum(0).view(-1)
        marginal_prob_s /= marginal_prob_s.sum()
        class_reg = math.log(marginal_prob.size(0)) + (marginal_prob * torch.log(marginal_prob)).sum() + math.log(
            marginal_prob_s.size(0)) + (marginal_prob_s * torch.log(marginal_prob_s)).sum()
        class_loss = self.ce(sim_cat, sim_labels) / N + class_reg
        return class_loss

def nta_loss(class_num, known_class_num, cluster_temperature, device, prob, prob_s):
    nta = NTA_Class(class_num, known_class_num, cluster_temperature, device).to(device)
    return nta(prob, prob_s)

def entropy_regularization(prob):
    EPS = 1e-8
    mean_prob = torch.mean(prob, 0)
    mean_prob_ = torch.clamp(mean_prob, min=EPS)
    prob_ent = mean_prob_ * torch.log(mean_prob_)
    prob_ent = - prob_ent.sum()
    return prob_ent

