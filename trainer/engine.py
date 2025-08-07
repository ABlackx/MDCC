import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from loaders.data_list import Pseudo_dataset
from loaders.target_loader import data_load
from model import network
from utils import loss as Loss
from utils.adjust_par import (adjust_consistency_weight, op_copy,
                            cosine_warmup)
from utils.tools import image_train, print_args
from utils.utils_noise import pair_selection_v1


class MDCC(object):
    def __init__(self, args):
        super(MDCC, self).__init__()

        self.encoder = network.encoder(args)
        self.encoder.load_model()
        self.netC = network.feat_classifier(type=args.layer, class_num=args.class_num,
                                            bottleneck_dim=args.bottleneck).cuda()
        self.is_vit = args.net.startswith(('vit', 'deit'))
        prefix = 'vit_' if self.is_vit else ''
        modelpath = os.path.join(args.output_dir_src, f'{prefix}source_C.pt')
        self.netC.load_state_dict(torch.load(modelpath))

        self.encoder = self.encoder.cuda()
        self.netC = self.netC.cuda()
        self.loader, self.dsets = data_load(args)
        self.max_iters = len(self.loader['two_train']) * args.max_epoch
        self.scaler = torch.cuda.amp.GradScaler()
        self.args = args
        self.ttransforms = image_train()

    def train_uns(self, epoch, adjust_learning_rate):
        for batchidx, (inputs, _, _, tar_idx) in enumerate(self.loader['two_train']):
            self.optimizer.zero_grad()
            if inputs.size(0) < 2:
                continue
            inputs = inputs.cuda()

            adjust_learning_rate(self.optimizer, (epoch - 1) * len(self.loader['two_train']) + batchidx + 1,
                                 self.max_iters,
                                 warmup_iters=self.args.scheduler_warmup_epochs * len(self.loader['two_train']))
            with autocast():
                features = self.encoder(inputs)
                outputs = self.netC(features)
                classifier_loss = torch.tensor(0.).cuda()

                softmax_out = nn.Softmax(dim=1)(outputs)
                entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                im_loss = entropy_loss * self.args.par_ent
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                im_loss -= gentropy_loss * self.args.par_ent
                classifier_loss += im_loss

            self.scaler.scale(classifier_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return classifier_loss.item()

    def train_su_cl(self, epoch, trainloader, trainSelloader, mem_label, initc, selected_pairs, selected_examples,
                    adjust_learning_rate):
        if trainSelloader:
            train_sel_iter = iter(trainSelloader)

        # tqdm
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader))

        for batchidx, (inputs, m_inputs, _, index) in progress_bar:
            # print(f"{batchidx} / {len(trainloader)}")

            progress_bar.set_description(f"Batch {batchidx + 1}/{len(trainloader)}")

            self.optimizer.zero_grad()
            if inputs.size(0) <= 1:
                continue
            pred = mem_label[index]
            bsz = inputs.size(0)
            inputs = inputs.cuda()  # image_train()
            m_inputs = m_inputs.cuda()  # mocov2()
            adjust_learning_rate(self.optimizer, (epoch - 1) * len(trainloader) + batchidx + 1, self.max_iters,
                                 warmup_iters=self.args.scheduler_warmup_epochs * len(trainloader))
            unselected_examples = torch.nonzero((selected_examples[index] == 0), as_tuple=True)[0]
            with autocast():
                features = self.encoder(inputs)
                # mix_features = self.mix_feature(features, pred, initc)
                outputs = self.netC(features)
                classifier_loss = torch.tensor(0.).cuda()
                if self.args.par_su_cl > 0:
                    q = outputs.clone()
                    features2 = self.encoder(m_inputs)
                    mix_features2 = self.mix_feature(features2, pred, initc)
                    k = self.netC(mix_features2)
                    q = F.normalize(q, dim=-1)
                    k = F.normalize(k, dim=-1)

                    embeds_batch = torch.cat([q, k], dim=0)  # (2 * bsz, cls_num)
                    pairwise_comp_batch = torch.matmul(embeds_batch, embeds_batch.t())  # 样本对之间在分类输出空间的相似性
                    maskSup_batch, maskUnsup_batch = self.mask_estimation(selected_pairs, index,
                                                                          bsz)  # mask
                    logits_mask_batch = (torch.ones_like(maskSup_batch) - torch.eye(2 * bsz).cuda())

                    loss_sup = self.args.par_su_cl * self.Supervised_ContrastiveLearning_loss(pairwise_comp_batch,
                                                                                              maskSup_batch,
                                                                                              maskUnsup_batch,
                                                                                              logits_mask_batch, bsz)
                else:
                    loss_sup = 0

                classifier_loss += loss_sup  # Loss_CL
                # consistency loss for M_2
                if len(unselected_examples) > 0 and self.args.par_consistency > 0:
                    weight = adjust_consistency_weight((epoch - 1) * len(trainloader) + batchidx + 1, self.max_iters,
                                                       warmup_iters=self.args.scheduler_warmup_epochs * len(
                                                           trainloader)) if self.args.das else 1.0
                    unselected_features = features[unselected_examples]
                    mixed_unselected_features = self.mix_feature(unselected_features, pred[unselected_examples], initc)
                    unselected_outputs = self.netC(mixed_unselected_features)
                    consistency_loss = self.kl_divergence(unselected_outputs, outputs[unselected_examples])
                    classifier_loss += consistency_loss * self.args.par_consistency * weight  # Loss_CS
                    # classifier_loss += consistency_loss * self.args.par_consistency # Loss_CS

                # Loss_CE * 系数
                if self.args.par_noisy_cls > 0:
                    if self.args.sel_cls:
                        assert trainSelloader is not None
                        try:
                            img, labels, _ = next(train_sel_iter)  # pseudo labels
                        except StopIteration:
                            train_sel_iter = iter(trainSelloader)
                            img, labels, _ = next(train_sel_iter)
                        img = img.cuda()
                        labels = labels.long().cuda()
                        sel_output = self.netC(self.encoder(img))
                        classifier_loss += nn.CrossEntropyLoss()(sel_output, labels) * self.args.par_noisy_cls
                    else:
                        cls_loss = nn.CrossEntropyLoss()(outputs, pred)
                        cls_loss *= self.args.par_noisy_cls
                        if epoch == 1 and self.args.dset == "VISDA-C":
                            cls_loss *= 0
                        classifier_loss += cls_loss

                if self.args.par_noisy_ent > 0:
                    softmax_out = nn.Softmax(dim=1)(outputs)
                    entropy_loss = torch.mean(Loss.Entropy(softmax_out))
                    im_loss = entropy_loss * self.args.par_noisy_ent
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.args.epsilon))
                    im_loss -= gentropy_loss * self.args.par_noisy_ent
                    classifier_loss += im_loss # Loss_IM

            self.scaler.scale(classifier_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return classifier_loss.item()

    def mask_estimation(self, selected_pairs, index, bsz):
        temp_graph = selected_pairs[index][:, index]  # 当前批次样本之间是否属于相同类别
        # Create mask without diagonal to avoid augmented view, i.e. this is supervised mask
        maskSup_batch = temp_graph.float().cuda()
        maskSup_batch[torch.eye(bsz) == 1] = 0  # 将对角线元素设置为 0，目的是移除每个样本与自己的关系
        maskSup_batch = maskSup_batch.repeat(2, 2)
        maskSup_batch[torch.eye(2 * bsz) == 1] = 0  # remove self-contrast case

        maskUnsup_batch = torch.eye(bsz, dtype=torch.float32).cuda()
        maskUnsup_batch = maskUnsup_batch.repeat(2, 2)
        maskUnsup_batch[
            torch.eye(2 * bsz) == 1] = 0  # remove self-contrast (weak-to-weak, strong-to-strong) case #2B*2B
        return maskSup_batch, maskUnsup_batch  # 正样本对和不同增强方法下的样本对

    def Supervised_ContrastiveLearning_loss(self, pairwise_comp, maskSup, maskUnsup, logits_mask, bsz):
        logits = torch.div(pairwise_comp, self.args.su_cl_t)
        exp_logits = torch.exp(logits) * logits_mask

        if self.args.scheduler_warmup_epochs == 1:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            # Approximation for numerical stability taken from supervised contrastive learning
        else:
            log_prob = torch.log(torch.exp(logits) + 1e-7) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos_sup = (maskSup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))
        mean_log_prob_pos_unsup = (maskUnsup * log_prob).sum(1) / (maskSup.sum(1) + maskUnsup.sum(1))

        lossa = -mean_log_prob_pos_unsup[:int(len(mean_log_prob_pos_unsup) / 2)] \
                - mean_log_prob_pos_sup[:int(len(mean_log_prob_pos_sup) / 2)]
        lossb = -mean_log_prob_pos_unsup[int(len(mean_log_prob_pos_unsup) / 2):] \
                - mean_log_prob_pos_sup[int(len(mean_log_prob_pos_sup) / 2):]
        loss = torch.cat((lossa, lossb))
        loss = loss.view(2, bsz).mean(dim=0)

        loss = ((maskSup[:bsz].sum(1)) > 0) * (loss.view(bsz))
        return loss.mean()

    def kl_divergence(self, p, q, epsilon=1e-6):
        # Apply softmax to ensure q is a valid probability distribution
        q = F.softmax(q, dim=-1) + epsilon
        p = F.softmax(p, dim=-1) + epsilon

        # Compute log-softmax for p
        log_p = torch.log(p + epsilon)

        # KL divergence in both directions, adding epsilon to avoid log(0)
        kl_pq = torch.sum(p * (log_p - torch.log(q + epsilon)), dim=-1)
        kl_qp = torch.sum(q * (torch.log(q + epsilon) - log_p), dim=-1)

        # Return mean of both KL divergences
        return torch.mean(0.5 * (kl_pq + kl_qp))

    def mix_feature(self, features, labels, initc, alpha=0.1, eps=1e-6):
        """
        Feature Mixing Module

        Args:
            features (Tensor): (N, feature_dim)
            labels (Tensor):  (N, )
            initc (Tensor): (class_num, feature_dim)
            alpha (float): control Beta distribution

        Returns:
            mixed_features (Tensor): (N, feature_dim)
        """
        mixed_features = features.clone()
        B, feature_dim = features.size()
        num_classes = initc.size(0)

        mu_x = features.mean(dim=1, keepdim=True)
        std_x = features.std(dim=1, keepdim=True)

        lmda = torch.distributions.Beta(alpha, alpha)
        lmda = lmda.sample((B, 1)).to(features.device)

        # Generate a mask to exclude the current sample's class center
        other_classes = torch.empty(labels.shape, dtype=torch.long, device=labels.device)

        for i in range(labels.size(0)):
            while True:
                other_class = torch.randint(0, num_classes, (1,), device=labels.device).item()
                if other_class != labels[i]:
                    other_classes[i] = other_class
                    break
        mixed_class_centers = initc[other_classes]  # shape: (B, feature_dim)

        # Compute mean and std of selected class centers
        mu_c = mixed_class_centers.mean(dim=1, keepdim=True)
        std_c = mixed_class_centers.std(dim=1, keepdim=True)

        # Calculate mixed mean and std for all samples
        mu_mix = lmda * mu_x + (1 - lmda) * mu_c
        std_mix = lmda * std_x + (1 - lmda) * std_c

        # Normalize and reconstruct feature for all samples
        normalized_features = (features - mu_x) / (std_x + 1e-6)
        mixed_features = normalized_features * std_mix + mu_mix

        return mixed_features.float()

    # def start_train(self):
    #     param_group = []
    #     for k, v in self.encoder.netF.named_parameters():
    #         if self.args.lr_decay1 > 0:
    #             if v.requires_grad:
    #                 param_group += [{'params': v, 'lr': self.args.lr * self.args.lr_decay1}]
    #         else:
    #             v.requires_grad = False
    #     for k, v in self.encoder.netB.named_parameters():
    #         if self.args.lr_decay2 > 0:
    #             if v.requires_grad:
    #                 param_group += [{'params': v, 'lr': self.args.lr * self.args.lr_decay2}]
    #         else:
    #             v.requires_grad = False
    #     for k, v in self.netC.named_parameters():
    #         v.requires_grad = False
    #
    #     optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    #     self.optimizer = op_copy(optimizer)
    #     acc_final = self.forward()
    #     return acc_final

    def start_train(self):
        param_group = []

        # 1. Backbone (netF)
        for k, v in self.encoder.netF.named_parameters():
            if self.args.lr_decay1 > 0:
                if v.requires_grad:
                    param_group += [{
                        'params': v,
                        'lr': self.args.lr * self.args.lr_decay1,
                        'weight_decay': 1e-4 if self.is_vit else 1e-3
                    }]
            else:
                v.requires_grad = False

        # 2. Bottleneck (netB)
        for k, v in self.encoder.netB.named_parameters():
            if self.args.lr_decay2 > 0:
                if v.requires_grad:
                    param_group += [{
                        'params': v,
                        'lr': self.args.lr * self.args.lr_decay2,
                        'weight_decay': 1e-4 if self.is_vit else 1e-3
                    }]
            else:
                v.requires_grad = False

        # 3. Classifier (netC)
        for k, v in self.netC.named_parameters():
            v.requires_grad = False

        # optimizer
        if self.is_vit:
            # ViT
            optimizer = optim.SGD(
                param_group,
                momentum=0.9,
                nesterov=True
            )
        else:
            # SGD + Momentum
            optimizer = optim.SGD(
                param_group,
                momentum=0.9,
                nesterov=True
            )

        # optimizer initiate
        self.optimizer = op_copy(optimizer)
        acc_final = self.forward()

        return acc_final

    def forward(self):
        best_acc = 0
        best_epoch = 0
        best_model_state = None

        for epoch in range(1, self.args.max_epoch + 1):
            self.encoder.eval()
            self.netC.eval()
            # 返回预测标签、处理后的特征、更新后的中心点、真实标签和原始输出。
            mem_label, all_fea, initc, all_label, all_output = self.obtain_label(False)

            mem_label = torch.from_numpy(mem_label).cuda()
            initc = torch.from_numpy(initc).cuda()
            self.encoder.train()
            self.netC.train()

            if epoch <= self.args.warmup_epochs:
                # when dataset == VisDA-C
                classifier_loss = self.train_uns(epoch, cosine_warmup)

            elif epoch > self.args.warmup_epochs:
                selected_examples, selected_pairs = pair_selection_v1(self.args.k_val,
                                                                      self.loader['test'],
                                                                      mem_label, self.args.class_num,
                                                                      self.args.cos_t,
                                                                      self.args.knn_times,
                                                                      all_fea,
                                                                      balance_class=self.args.balance_class,
                                                                      sel_ratio=self.args.sel_ratio,
                                                                      )

                # calculate pseudo-label accuracy of selected_examples
                self.encoder.eval()
                self.netC.eval()
                self.cal_sel_acc(real_labels=all_label, mem_labels=mem_label,
                                 selected_samples=selected_examples)
                self.cal_all_acc(real_labels=all_label, mem_labels=mem_label)

                # use the selected pseudo-labels to build a dataloader train_sel_loader to supervise training
                self.encoder.train()
                self.netC.train()
                txt_tar = open(self.args.t_dset_path).readlines()
                pseudo_dataset = Pseudo_dataset(txt_tar, mem_label.cpu().numpy(), transform=self.ttransforms,
                                                append_root=self.args.append_root)
                train_sel_loader = DataLoader(pseudo_dataset, batch_size=self.args.batch_size,
                                              num_workers=self.args.worker,
                                              pin_memory=True,
                                              sampler=torch.utils.data.WeightedRandomSampler(selected_examples,
                                                                                             len(selected_examples)))

                classifier_loss = self.train_su_cl(epoch, self.loader['two_train'], train_sel_loader,
                                                   mem_label, initc, selected_pairs, selected_examples, cosine_warmup)

            # evaluate accuracy every epoch
            self.encoder.eval()
            self.netC.eval()
            if self.args.dset == 'VISDA-C':
                acc_s_te, acc_list = self.cal_acc(True)
                log_str = f'Task: {self.args.name}, epoch:{epoch}/{self.args.max_epoch}; Accuracy = {acc_s_te:.2f};' \
                          f'Loss = {classifier_loss:.2f}; \n {acc_list}'
            else:
                acc_s_te = self.cal_acc(False)
                log_str = f'Task: {self.args.name}, epoch:{epoch}/{self.args.max_epoch}; Accuracy = {acc_s_te:.2f} ;' \
                          f'Loss = {classifier_loss:.2f} '
            self.args.out_file.write(log_str + '\n')
            self.args.out_file.flush()
            print(log_str + '\n')

            # update best
            if acc_s_te > best_acc:
                best_acc = acc_s_te
                best_epoch = epoch
                best_model_state = {
                    'encoder_F': self.encoder.netF.state_dict(),
                    'encoder_B': self.encoder.netB.state_dict(),
                    'netC': self.netC.state_dict()
                }
        log_str = f'Task: {self.args.name}, best epoch:{best_epoch}; Accuracy = {best_acc:.2f}%'
        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        print(log_str + '\n')
        # save best
        if self.args.issave:
            self.args.save_dir = osp.join(self.args.output_dir,
                                          f'acc_{best_acc:.2f}_{self.args.savename}_epoch_{best_epoch}')
            # if not osp.exists(self.args.save_dir):
            #     os.system('mkdir -p ' + self.args.save_dir)
            if not osp.exists(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            prefix = 'vit_' if self.is_vit else ''
            torch.save(best_model_state['encoder_F'],
                       osp.join(self.args.save_dir, f"{prefix}target_F.pt"))
            torch.save(best_model_state['encoder_B'],
                       osp.join(self.args.save_dir, f"{prefix}target_B.pt"))
            torch.save(best_model_state['netC'],
                       osp.join(self.args.save_dir, f"{prefix}target_C.pt"))
        return round(best_acc, 2)

    def cal_sel_acc(self, real_labels, mem_labels, selected_samples):
        # accuracy of selected samples
        with torch.no_grad():
            idx_selected = selected_samples.nonzero().squeeze()
            sel_mem_labels = mem_labels[idx_selected]
            sel_real_labels = real_labels[idx_selected]
            sel_acc = (sel_real_labels == sel_mem_labels).sum().item() / selected_samples.sum().item()
        logstr = f'selection samples accuracy:{100 * sel_acc:.2f}%'
        print(logstr)
        self.args.out_file.write(logstr + '\n')
        self.args.out_file.flush()

    def cal_all_acc(self, real_labels, mem_labels):
        # accuracy of all pesudo labels
        with torch.no_grad():
            total = real_labels.size(0)
            correct = (real_labels == mem_labels).sum().item()
            acc = correct / total
        logstr = f'all pseudo labels accuracy: {100 * acc:.2f}%'
        print(logstr)
        self.args.out_file.write(logstr + '\n')
        self.args.out_file.flush()

    def cal_acc(self, flag=False):
        start_test = True
        with torch.no_grad():
            iter_train = iter(self.loader['test'])
            for i in range(len(self.loader['test'])):
                data = next(iter_train)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                outputs = self.netC(self.encoder(inputs))
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat([all_output, outputs.float().cpu()], 0)
                    all_label = torch.cat([all_label, labels.float()], 0)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        if flag:
            matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
            acc = matrix.diagonal() / matrix.sum(axis=1) * 100  # each class
            aacc = acc.mean()  # mean each class
            aa = [str(np.round(i, 2)) for i in acc]
            acc = ' '.join(aa)  # "90.91 90.00 80.00"
            return aacc, acc
        else:
            return accuracy * 100

    def obtain_label(self, return_dist=False):
        start_test = True
        with torch.no_grad():
            iter_train = iter(self.loader['test'])
            for _ in range(len(self.loader['test'])):
                data = next(iter_train)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                feas = self.encoder(inputs)
                outputs = self.netC(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat([all_fea, feas.float().cpu()], 0)
                    all_output = torch.cat([all_output, outputs.float().cpu()], 0)
                    all_label = torch.cat([all_label, labels.float()], 0)
        all_output = nn.Softmax(dim=1)(all_output)  # N*C
        _, predict = torch.max(all_output, 1)  # N*1

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        if self.args.distance == 'cosine':  # 添加一列全1并归一化
            all_fea = torch.cat([all_fea, torch.ones(all_fea.size(0), 1)], 1)  # N*(K+1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()  # 行归一化
            all_fea = all_fea.float().cpu().numpy()  # 行代表样本，列代表K维特征维度上的特征

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()  # 行代表样本，列代表每个类的概率分数
        initc = aff.transpose().dot(all_fea)  # 输出矩阵转置后与特征矩阵相乘，每一行可以看作是对应类别的初始中心点（在特征空间中）
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # 再除以aff（每个类别的所有样本的和），得到归一化的初始中心点
        cls_count = np.eye(K)[predict].sum(
            axis=0)  # N*K -> 1*K 生成一个单位矩阵，将 predict 中的每个预测标签转换为one-hot编码。然后对one-hot编码的矩阵按列求和，得到每个类别在预测结果中出现的次数。
        labelset = np.where(cls_count > 0)
        labelset = labelset[0]  # 得到预测结果中出现过的类别索引

        dd = cdist(all_fea, initc[labelset], self.args.distance)  # 计算每个样本与出现过的所有类别中心间的余弦相似度
        pred_label = dd.argmin(axis=1)  # dd-->(n_samples, n_centers) 返回每个样本最靠近的样本中心的索引
        pred_label = labelset[pred_label]  # 按索引值到标签集中取得标签数字，标签集是出现过的类别，不全，因此需由索引取到真实标签

        for round in range(1):  # 进行一次迭代，更新类别中心，再更新标签
            aff = np.eye(K)[pred_label]  # 基于新的预测标签生成one-hot矩阵，表示每个样本属于哪个类别
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc[labelset], self.args.distance)
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]

        min_dist = dd.min(axis=1)  # 最小距离
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = f'Task:{self.args.name}   Accuracy = {accuracy * 100:.2f}% -> {acc * 100:.2f}%'

        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        print(log_str + '\n')

        if return_dist:
            return pred_label.astype('int'), min_dist
        else:
            return pred_label.astype('int'), torch.from_numpy(all_fea[:, :-1]).cuda(), initc[:,
                                                                                       :-1], all_label.float().cuda(), all_output.cuda()
