import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
from utils import set_random_seed,cluster_acc, accuracy, AverageMeter, representation_constraint, Centroid_Constraint_Class
from utils import debiasing_regularization, initialize_beta, update_beta, nta_loss, entropy_regularization
from open_world_cifar import get_dataset_class

def train(args, model,  device, train_label_loader, train_unlabel_loader, optimizer, epoch, tf_writer, beta, known_novel_gap):
    model.train()
    bce = nn.BCELoss()
    centroid_constraint = Centroid_Constraint_Class(num_classes=args.no_class, feature_dim=512).to(device)

    unlabel_loader_iter = cycle(train_unlabel_loader)
    known_probs = AverageMeter('known_prob', ':.4e')
    novel_probs = AverageMeter('novel_prob', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    sup_losses = AverageMeter('sup_losses', ':.4e')
    debiased_losses = AverageMeter('debiased_losses', ':.4e')
    instance_losses = AverageMeter('instance_losses', ':.4e')
    class_losses = AverageMeter('class_losses', ':.4e')

    for batch_idx, ((x, x2, x3), target, _,temp_l) in enumerate(train_label_loader):
        ((ux, ux2, ux3),u_target,_,temp_u) = next(unlabel_loader_iter)
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        x3 = torch.cat([x3, ux3], 0)
        x, x2, x3, target = x.to(device), x2.to(device), x3.to(device), target.to(device)
        optimizer.zero_grad()

        logits, feat = model(x)
        logits2, feat2 = model(x2)
        logits3, feat3 = model(x3)
        labeled_len = len(target)
        logits_uw = logits2[labeled_len:]
        logits_us = logits3[labeled_len:]
        prob = F.softmax(logits, dim=1)
        prob2 = F.softmax(logits2, dim=1)
        prob3 = F.softmax(logits3, dim=1)

        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        feat_dist = torch.mm(feat_norm, feat_norm.t())
        instance_pairs = []
        target_np = target.cpu().numpy()
        labeled_feat_dist = feat_dist[:labeled_len, :labeled_len]

        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                instance_pairs.append(idxs[0])
            else:
                labeled_feat_dist_np = labeled_feat_dist.cpu().numpy()
                feat_dist_row = labeled_feat_dist_np[i]
                if len(idxs) < 1:
                    min_index = np.argmax(feat_dist_i_sameClass)
                else:
                    feat_dist_i_sameClass = feat_dist_row[idxs]
                    min_index = np.argmin(feat_dist_i_sameClass)
                min_idxs = idxs[min_index]
                selec_idx = min_idxs
                instance_pairs.append(int(selec_idx))
        unlabeled_feat_dist = feat_dist[labeled_len:, :]
        vals, matched_idx = torch.topk(unlabeled_feat_dist, 2, dim=1)
        max_matched = torch.topk(feat_dist[:labeled_len, matched_idx[:, 1]], args.delta_sim_order ,dim = 0)[0][args.delta_sim_order - 1]
        mask_1 = (vals[:, 1] - max_matched).ge(0).float()
        mask_0 = (vals[:, 1] - max_matched).lt(0).float()
        matched_idx_1 = (matched_idx[:, 1] * mask_1).cpu().numpy()
        matched_idx_0 = (matched_idx[:, 0] * mask_0).cpu().numpy()
        matched_idx = (matched_idx_1 + matched_idx_0).flatten().tolist()
        instance_pairs.extend(matched_idx)

        matched_prob = prob2[instance_pairs, :]
        matched_logits = logits2[instance_pairs, :]
        prob_row = prob.view(args.batch_size, 1, -1)
        matched_prob_col = matched_prob.view(args.batch_size, -1, 1)
        prob_w = torch.softmax(logits.detach(), dim=-1)
        prob_w_matched = torch.softmax(matched_logits, dim=-1)
        matched_feats = feat2[instance_pairs, :]
        rc_loss = representation_constraint(prob_row, matched_prob_col, prob_w_matched, prob_w, feat2,
                                            matched_feats, centroid_constraint, bce, args.rc_flag, epoch)

        split_class = int(args.no_seen)
        pseudo_label = debiasing_regularization(logits_uw.detach(), beta,  args.eta, known_novel_gap, split_class)
        max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.delta_p).float()
        index_known = pseudo_targets_u.lt(split_class).float()
        index_novel = pseudo_targets_u.ge(split_class).float()
        known_count = torch.sum(index_known)
        novel_count = torch.sum(index_novel)
        if known_count > 0:
            known_prob = torch.sum(index_known * max_probs) / known_count + 1e-10
        else:
            known_prob = torch.tensor(0.0, device=device)
        if novel_count > 0:
            novel_prob = torch.sum(index_novel * max_probs) / novel_count + 1e-10
        else:
            novel_prob = torch.tensor(0.0, device=device)
        beta_mask = mask if args.masked_beta else None
        beta = update_beta(torch.softmax(logits_uw.detach(), dim=-1), beta, momentum=args.epsilon, beta_mask = beta_mask)

        tau_known = args.eta - known_novel_gap
        tau_novel = args.eta + known_novel_gap
        delta_logits = torch.log(beta)
        delta_logits[:, :split_class] *= tau_known
        delta_logits[:, split_class:] *= tau_novel
        logits_us = logits_us - delta_logits
        debiased_loss = (F.cross_entropy(logits_us, pseudo_targets_u, reduction='none', weight=None) * mask).mean()

        sup_loss = (F.cross_entropy(logits[:labeled_len], target.long(), reduction='none')).mean()
        instance_loss = rc_loss + sup_loss + debiased_loss

        prob1 = F.softmax(logits, dim=1)
        class_loss = nta_loss(args.no_class, args.no_seen, args.nta_temperature, device, prob1, prob3)
        entropy_loss = entropy_regularization(prob)
        total_loss = args.lambda_instance * instance_loss + args.lambda_class * class_loss - entropy_loss

        known_probs.update(known_prob.item(),args.batch_size)
        novel_probs.update(novel_prob.item(),args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        sup_losses.update(sup_loss.item(), args.batch_size)
        debiased_losses.update(debiased_loss.item(), args.batch_size)
        instance_losses.update(instance_loss.item(),args.batch_size)
        class_losses.update(class_loss.item(),args.batch_size)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    known_novel_gap = known_probs.avg - novel_probs.avg
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)
    tf_writer.add_scalar('loss/sup_loss', sup_losses.avg, epoch)
    tf_writer.add_scalar('loss/debiased_loss', debiased_losses.avg, epoch)
    tf_writer.add_scalar('loss/instance_loss', instance_losses.avg, epoch)
    tf_writer.add_scalar('loss/class_loss', class_losses.avg, epoch)
    return beta, known_novel_gap

def test(args, model, known_class_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    probs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            logits, _, = model(x)
            prob = F.softmax(logits, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            probs = np.append(probs, prob.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    seen_mask = targets < known_class_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    print('Epoch: [{}/{}], Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(epoch, args.epochs, overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)

def main():
    parser = argparse.ArgumentParser(description='OpenHRD')
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--data-root', default=f'datasets', help='directory to store data')
    parser.add_argument('--lbl-percent', type=int, default=50, help='Percentage of labeled data')
    parser.add_argument('--novel-percent', type=int, default=50, help='Percentag of novel classes, default 50')
    parser.add_argument("--imb-factor", default=1, type=float, help="imbalance factor of the data, default 1")
    parser.add_argument('--seed', type=int, default=3, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str,default='name')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('-b', '--batch-size', type=int, default=512, metavar='N',help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument('--nta_temperature', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--rc_flag', type=str, default='func',help='rc implementation: func or class')
    parser.add_argument('--lambda_instance', type=float, default=1.0)
    parser.add_argument('--lambda_class', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.999)
    parser.add_argument('--delta_p', type=float, default=0.5)
    parser.add_argument('--delta_sim_order', type=int, default=2)
    parser.add_argument('--masked_beta', action='store_true', default=True)
    parser.add_argument('--initial_gap', type=float, default=0.01)
    args = parser.parse_args()
    set_random_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.name += f"-{args.dataset}"
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if args.dataset == "cifar10":
        args.no_class = 10
    elif args.dataset == "cifar100":
        args.no_class = 100
    elif args.dataset == "imagenet100":
        args.no_class = 100
    args.no_seen = args.no_class - int((args.novel_percent*args.no_class)/100)
    dataset_class = get_dataset_class(args)
    train_label_set, train_unlabel_set,test_set, test_dataset_seen, test_dataset_novel = dataset_class.get_dataset()
    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set,batch_size=args.batch_size - labeled_batch_size, shuffle=True,
                                                       num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.num_workers)
    model = models.resnet18(num_classes=args.no_class)
    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    tf_writer = SummaryWriter(log_dir=args.savedir)
    beta = initialize_beta(class_num=args.no_class)
    known_novel_gap = args.initial_gap
    for epoch in range(args.epochs):
        beta,known_novel_gap = train(args, model, device, train_label_loader, train_unlabel_loader, optimizer,epoch, tf_writer, beta, known_novel_gap)
        test(args, model, args.no_seen, device, test_loader, epoch, tf_writer)
        scheduler.step()
    tf_writer.close()

if __name__ == '__main__':
    main()
