# from __future__ import print_function
import torch
import warnings
warnings.filterwarnings('ignore')


def pair_selection_v1(k_val, test_loader, labels, class_num, cos_t, knn_times, train_features, balance_class=True,
                   sel_ratio=0, corrected=False):
    """
    k_val:  neighbors number of knn
    labels: pseudo-labels obtained from feature prototypes
    """
    # 存储所有样本之间的相似性度量（由特征之间的相似性构成）
    similarity_graph_all = torch.zeros(len(test_loader.dataset), len(test_loader.dataset))
    # 训练样本的噪声标签（克隆自伪标签）
    train_noisy_labels = labels.clone().long().cuda()
    # 伪标签，用于在迭代过程中更新。
    train_labels = labels.clone().long().cuda()
    # 衡量每个样本的预测不确定度。
    discrepancy_measure = torch.zeros((len(test_loader.dataset),)).cuda()
    # 衡量每个样本的预测是否与伪标签一致
    agreement_measure = torch.zeros((len(test_loader.dataset),))

    with torch.no_grad():
        for i in range(knn_times):
            print(f'starting the {i+1}st knn....')
            retrieval_one_hot_train = torch.zeros(k_val, class_num).cuda()  # 邻居数*类别数，表示每个邻居属于何种类别
            train_new_labels = train_labels.clone()  # 克隆伪标签
            for batch_idx, (_, _, index) in enumerate(test_loader):
                batch_size = index.size(0)
                features = train_features[index]  # N*K特征矩阵中取出对应的批次的样本的特征

                # similarity graph
                dist = torch.mm(features, train_features.t())  # 计算样本之间的内积（点积），衡量样本的相似度
                similarity_graph_all[index] = dist.cpu().detach()  # 用于累积所有测试批次的结果，转移至cpu且与计算图分离
                dist[torch.arange(dist.size(0)), index] = -1  # 将自身相似性设为 -1, 如批次大小为2，index为3，4时，则[[?,?,?,-1,?,···], [?,?,?,?,-1,···]] 2*N

                # sample k-nearest neighbors
                yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  # 每行的最近邻距离值/最近邻的索引（列索引）
                candidates = train_new_labels.view(1, -1).expand(batch_size, -1)  # bs*N, 每行都是所有样本的伪标签
                retrieval = torch.gather(candidates, 1, yi)  # retrieval 的形状是 (batch_size, k_val)，包含了每个测试样本的 k 个最近邻的标签

                retrieval_one_hot_train.resize_(batch_size * k_val, class_num).zero_()
                # scatter_操作将retrieval中的每个索引（现在被展平为(-1, 1)形状）映射到retrieval_one_hot_train的第二维（类别维）上，并将这些位置的值设置为1
                # retrieval.view(-1, 1) = [[1], [2], [0], [3], [1], [2]]
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1).long(), 1)  # 生成独热编码，(bs*k_val) * cls

                yd_transform = torch.exp(yd.clone().div_(cos_t))  # 将距离转换为权重，使得较近的邻居具有更高的权重
                # 逐元素乘距离作为权重
                '''
                [[[0, 8103.08, 0, 0],  # 第一个样本的第一个邻居加权标签
                  [0, 0, 1096.63, 0],  # 第一个样本的第二个邻居加权标签
                  [2980.96, 0, 0, 0]], # 第一个样本的第三个邻居加权标签
                 [[0, 0, 0, 148.41],   # 第二个样本的第一个邻居加权标签
                  [0, 403.43, 0, 0],   # 第二个样本的第二个邻居加权标签
                  [0, 0, 54.6, 0]]]    # 第二个样本的第三个邻居加权标签
                  
                probs_corrected = [[2980.96, 8103.08, 1096.63, 0],  # 第一个样本的类别加权和
                                    [0, 403.43, 54.6, 148.41]]      # 第二个样本的类别加权和
                probs_norm: 每个样本属于各个类别的概率分布
                '''
                probs_corrected = torch.sum(torch.mul(retrieval_one_hot_train.view(batch_size, -1, class_num),
                                                      yd_transform.view(batch_size, -1, 1)), 1)
                probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

                # 每个样本的伪标签的概率，概率对应于上述根据最近邻样本距离加权投票得到，并非分类器得到的
                prob_temp = probs_norm[torch.arange(0, batch_size), labels[index]]
                prob_temp[prob_temp <= 1e-4] = 1e-4
                prob_temp[prob_temp > (1 - 1e-4)] = 1 - 1e-4
                discrepancy_measure[index] = -torch.log(prob_temp)  # 每个样本的预测标签的不确定度计算，概率越接近0，不确定度越高；概率越接近1，不确定度越低

                # update the labels
                sorted_pro, predictions_corrected = probs_norm.sort(1, True)  # 降序排列。最大概率，对应的标签索引
                new_labels = predictions_corrected[:, 0]
                train_labels[index] = new_labels  # 每个样本概率最大的类别标签作为新的预测标签
                agreement_measure[index.data.cpu()] = (torch.max(probs_norm, dim=1)[1] == labels[index]).float().data.cpu()  # 对于index指定的样本，比较模型的预测和原始伪标签，一致为1.0，否则为0.0
            selected_examples = agreement_measure  # 协议度量，表示某个样本是否预测正确


    if balance_class:
        # select the top k_corrected samples for each class
        agreement_measure = torch.zeros((len(labels),)).cuda()
        for i in range(class_num):
            idx_class = labels == i
            num_per_class = idx_class.sum()  # i类别的样本数量
            idx_class = (idx_class.float() == 1.0).float().nonzero().squeeze()  # 类别 i 样本的索引张量
            discrepancy_class = discrepancy_measure[idx_class]  # 取i类样本的预测的不确定度

            k_corrected = sel_ratio * num_per_class  # 要选择的样本数
            if k_corrected >= 1:
                # 从类别 i 的样本中选出不确定度最小的 k_corrected 个样本，返回相对索引
                # largest=False,升序
                top_clean_class_relative_idx = \
                    torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=True)[1]

                i_sel_index = idx_class[top_clean_class_relative_idx]
                agreement_measure[i_sel_index] = 1.0  # 将选中的样本在 agreement_measure 中对应位置设置为 1.0，表示这些样本被选出
                if corrected:  # 若为True，进一步选择不确定度最高的样本，确保边界样本也被选中
                    the_val = discrepancy_class[top_clean_class_relative_idx[-1]]
                    the_val_index = (discrepancy_class == the_val).float().nonzero().squeeze().long()
                    agreement_measure[idx_class[the_val_index]] = 1.0
        selected_examples = agreement_measure

    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()  # 所有选中的样本索引
        total_selected_num = len(index_selected)
        train_noisy_labels = train_noisy_labels.cpu().unsqueeze(1)  # 初始伪标签展开为N*1

        noisy_pairs = torch.eq(train_noisy_labels, train_noisy_labels.t())  # 对称的布尔矩阵，表示哪些样本具有相同的伪标签
        final_selected_pairs = torch.zeros_like(noisy_pairs).type(torch.uint8)

        selected_pairs = noisy_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)].clone()
        final_selected_pairs[
            index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num), index_selected.unsqueeze(
                0).expand(total_selected_num, total_selected_num)] = selected_pairs.type(torch.uint8)
        final_selected_pairs = final_selected_pairs.type(torch.bool)
    return selected_examples.cuda(), final_selected_pairs.contiguous()
