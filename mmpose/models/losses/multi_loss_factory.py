# ------------------------------------------------------------------------------
# Adapted from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
# Original licence: Copyright (c) Microsoft, under the MIT License.
# Modified by Congju Du (ducongju@hust.edu.cn).
# ------------------------------------------------------------------------------

from math import factorial
from cv2 import pencilSketch
import numpy as np
import torch
import torch.nn as nn

from ..builder import LOSSES


def _make_input(t, requires_grad=False, device=torch.device('cpu')):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device

    Returns:
        torch.Tensor: zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)
    return inp


@LOSSES.register_module()
class HeatmapLoss(nn.Module):
    """Accumulate the heatmap loss for each image in the batch.

    Args:
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self, supervise_empty=True):
        super().__init__()
        self.supervise_empty = supervise_empty

    def forward(self, pred, gt, mask):
        """Forward function.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred (torch.Tensor[N,K,H,W]):heatmap of output.
            gt (torch.Tensor[N,K,H,W]): target heatmap.
            mask (torch.Tensor[N,H,W]): mask of target.
        """
        assert pred.size() == gt.size(
        ), f'pred.size() is {pred.size()}, gt.size() is {gt.size()}'

        if not self.supervise_empty:
            empty_mask = (gt.sum(dim=[2, 3], keepdim=True) > 0).float()
            loss = ((pred - gt)**2) * empty_mask.expand_as(
                pred) * mask[:, None, :, :].expand_as(pred)
        else:
            loss = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


@LOSSES.register_module()
class AELoss(nn.Module):
    """Associative Embedding loss.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`_.
    """

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred_tag (torch.Tensor[KxHxW,1]): tag of output for one image.
            joints (torch.Tensor[M,K,2]): joints information for one image.
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, joints):
        """Accumulate the tag loss for each image in the batch.

        Note:
            batch_size: N
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
            num_keypoints: K

        Args:
            tags (torch.Tensor[Nx(KxHxW)x1]): tag channels of output.
            joints (torch.Tensor[NxMxKx2]): joints information.
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


@LOSSES.register_module()
class AEgroupLoss(nn.Module):
    def __init__(self, loss_type, o_pull_weight, l_pull_weight, h_pull_weight, group_type):
        super().__init__()
        self.loss_type = loss_type
        self.group_type = group_type
        self.o_pull_weight = o_pull_weight
        self.l_pull_weight = l_pull_weight
        self.h_pull_weight = h_pull_weight

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        joints: num_people, num_joints, 2 
        shape = (30, 17, 2)
        pred_tag: w * h * num_joints, 1
        shape = torch.Size([278528, 1]) = 128 * 128 * 17
        """

        tags = []
        group_tags = []
        
        # pulling loss
        if self.group_type == 'hierarchical_coco':
            groups = [
                [0,1,2,3,4,5,6],[11,12],[7],[8],[9],[10],[13],[14],[15],[16]
            ]   
        elif self.group_type == 'hierarchical_crowdpose':  # 论文方案
            groups = [
                [0,1,12,13],[6,7],[2],[3],[4],[5],[8],[9],[10],[11]
            ]
        elif self.group_type == 'spatial_prior_crowdpose':  # 论文中的空间先验
            groups = [
                [12,13],[0,1,6,7],[2,4],[3,5],[8,10],[9,11]
            ]
        elif self.group_type == 'mutual_information_crowdpose':  # 论文中的空间先验
            groups = [
                [0,1,12,13],[6,7],[2,4],[3,5],[8,10],[9,11]
            ]
        elif self.group_type == 'test':
            groups = [
                [0,1],[2,3]
            ]
        else:
            raise ValueError('Unkown ae group type')

        o_pull = 0
        l_pull = 0
        h_pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:  # 如果向量位置（非坐标位置）对应有关节点，那么导出这个位置
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)  # 沿着一个新维度对输入张量序列进行连接
            tags.append(torch.mean(tmp, dim=0))  # 最后一维是某个人的平均tag值

            if self.o_pull_weight != 0.0:
                o_pull = o_pull + len(joints_per_person) * torch.mean((tmp - tags[-1].expand_as(tmp))**2)

            for group_index in groups:  # 组数求和
                tmp2 = []
                for joints_index in group_index:  # 组内关节数求和
                    if joints_per_person[joints_index][1] > 0:
                        tmp2.append(pred_tag[joints_per_person[joints_index][0]])
                if len(tmp2) == 0:
                    continue
                tmp2 = torch.stack(tmp2) # dim=0拼接
                group_tags.append(torch.mean(tmp2, dim=0))  # 最后一维是一个组的平均tag值
                l_pull = l_pull + len(group_index) * torch.mean((tmp2 - group_tags[-1].expand_as(tmp2))**2)  # 一个组的pull loss
                h_pull = h_pull + ((tags[-1] - group_tags[-1])**2).squeeze(0)  # 组与人的pull loss

        num_tags = len(tags)
        num_groups = len(groups)
        # 没有人有可视关节，返回0,0
        if num_tags == 0:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), \
                _make_input(torch.zeros(1).float(), device=pred_tag.device)
        # 只有一个人有可视关节，push loss返回0 tensor
        elif num_tags == 1 and self.o_pull_weight != 0.0:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), \
                (o_pull/(num_tags) * self.o_pull_weight + l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)
        elif num_tags == 1 and self.o_pull_weight == 0.0:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), \
                (l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)

        # pushing loss
        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)  # 对调tensor维度

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)  # 对差逐元素平方
            push = torch.exp(-diff)  # 转换成指数形式
            push = torch.sum(push) - num_tags  # 归一化项
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        if self.o_pull_weight != 0.0:
            return push/((num_tags - 1) * num_tags) * 0.5, \
                (o_pull/(num_tags) * self.o_pull_weight + l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)
        
        if self.o_pull_weight == 0.0:
            return push/((num_tags - 1) * num_tags) * 0.5, \
                (l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)
        

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


@LOSSES.register_module()
class AEgroupfocalLoss(nn.Module):
    """Associative Embedding loss.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`
    """

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
            num_keypoints: K

        Args:
            pred_tag (torch.Tensor[(KxHxW)x1]): tag of output for one image.
            joints (torch.Tensor[MxKx2]): joints information for one image.
        """

        import numpy as np
        import math

        tags = []
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])  # 存放当前人体实例的tag值
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))  # 存放所有人体实例的tag均值

            # IQR四分位距算法
            tmp2 = tmp.cpu().detach().numpy()
            tmp2 = sorted(tmp2)
            lower_q = np.quantile(tmp2, 0.25, interpolation='lower')  # 下四分位数
            higher_q = np.quantile(tmp2, 0.75, interpolation='higher')  # 上四分位数
            int_r = higher_q - lower_q  # 四分位距

            factor = []
            for i in range(len(tmp)):                
                if tmp2[i] < lower_q - 1.5 * int_r or tmp2[i] > higher_q + 1.5 * int_r:
                    factor.append(((torch.Tensor([2]).to(pred_tag.device).to(pred_tag.device))))
                else:
                    factor.append(torch.Tensor([1]).to(pred_tag.device))
            factor = torch.stack(factor)
)
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2*factor)

        num_tags = len(tags)
        
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp_focal':
            diff_mean = torch.mean(abs(diff))
            factor_push = torch.zeros(num_tags, num_tags).to(pred_tag.device)

            for i in range(num_tags):
                for j in range(num_tags):
                    if abs(diff[i,j]) < diff_mean and abs(diff[i,j]) != 0:
                        factor_push[i,j] = 1.2
                    elif abs(diff[i,j]) == 0:
                        factor_push[i,j] = 1.0
                    else:
                        factor_push[i,j] = 0.8

            diff = torch.pow(diff, 2)
            push = torch.exp(-diff) * factor_push
            push = torch.sum(push) - num_tags
        
        elif self.loss_type == 'exp_normal':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        
        elif self.loss_type == 'exp_topk':
            diff = abs(diff)
            diff = torch.reshape(diff, (-1,1))
            diff_after = []
            for index, value in enumerate(diff):
	            if(value != 0):
		            diff_after.append(diff[index])
            diff_after2 = torch.tensor(diff_after).to(pred_tag.device)
            
            diff, _ = torch.topk(diff_after2, k=math.ceil(0.7*diff_after2.size(0)), dim=0)
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push)  # 0值通过减去num_tags抵消了，所以这里不需要再去删除了

        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags

        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5  # 因为矩阵乘了两遍
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, joints):
        """Accumulate the tag loss for each image in the batch.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            tags (torch.Tensor[N,KxHxW,1]): tag channels of output.
            joints (torch.Tensor[N,M,K,2]): joints information.
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


@LOSSES.register_module()
class AEgroupfocalLossv2(nn.Module):
    def __init__(self, loss_type, o_pull_weight, l_pull_weight, h_pull_weight, group_type):
        super().__init__()
        self.loss_type = loss_type
        self.group_type = group_type
        self.o_pull_weight = o_pull_weight
        self.l_pull_weight = l_pull_weight
        self.h_pull_weight = h_pull_weight

    def singleTagLoss(self, pred_tag, joints):
        """
        associative embedding loss for one image
        joints: num_people, num_joints, 2 
        shape = (30, 17, 2)
        pred_tag: w * h * num_joints, 1
        shape = torch.Size([278528, 1]) = 128 * 128 * 17
        """

        tags = []
        group_tags = []
        
        # pull loss
        if self.group_type == 'hierarchical_crowdpose':
            groups = [
                [0,1,12,13],[6,7],[2],[3],[4],[5],[8],[9],[10],[11]
            ]
        else:
            raise ValueError('Unkown ae group type')

        o_pull = 0
        l_pull = 0
        h_pull = 0
        for joints_per_person in joints:  # 人数求和
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:  # 如果向量位置（非坐标位置）对应有关节点，那么导出这个位置
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)  # 沿着一个新维度对输入张量序列进行连接
            tags.append(torch.mean(tmp, dim=0))  # 最后一维是某个人的平均tag值

            if self.o_pull_weight != 0.0:
                o_pull = o_pull + len(joints_per_person) * torch.mean((tmp - tags[-1].expand_as(tmp))**2*factor)

            for group_index in groups:  # 组数求和
                tmp2 = []
                for joints_index in group_index:  # 组内关节数求和
                    if joints_per_person[joints_index][1] > 0:
                        tmp2.append(pred_tag[joints_per_person[joints_index][0]])
                if len(tmp2) == 0:
                    continue
                tmp2 = torch.stack(tmp2) # dim=0拼接
                group_tags.append(torch.mean(tmp2, dim=0))  # 最后一维是一个组的平均tag值

                import numpy as np
                tmp3 = tmp2.cpu().detach().numpy()
                tmp3 = sorted(tmp3)
                lower_q = np.quantile(tmp3, 0.25, interpolation='lower')  # 下四分位数
                higher_q = np.quantile(tmp3, 0.75, interpolation='higher')  # 上四分位数
                int_r = higher_q - lower_q  # 四分位距

                factor = []
                for i in range(len(tmp2)):                
                    if tmp3[i] < lower_q - 1.5 * int_r or tmp3[i] > higher_q + 1.5 * int_r:
                        factor.append(((torch.Tensor([2]).to(pred_tag.device).to(pred_tag.device))))
                    else:
                        factor.append(torch.Tensor([1]).to(pred_tag.device))
                factor = torch.stack(factor)    

                l_pull = l_pull + len(group_index) * torch.mean((tmp2 - group_tags[-1].expand_as(tmp2))**2*factor)  # 一个组的pull loss
                h_pull = h_pull + ((tags[-1] - group_tags[-1])**2).squeeze(0)  # 组与人的pull loss

        num_tags = len(tags)
        num_groups = len(groups)
        # 没有人有可视关节，返回0,0
        if num_tags == 0:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), \
                _make_input(torch.zeros(1).float(), device=pred_tag.device)
        # 只有一个人有可视关节，push loss返回0 tensor
        elif num_tags == 1 and self.o_pull_weight != 0.0:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), \
                (o_pull/(num_tags) * self.o_pull_weight + l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)
        elif num_tags == 1 and self.o_pull_weight == 0.0:
            return _make_input(torch.zeros(1).float(), device=pred_tag.device), \
                (l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)

        # push loss
        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)  # 对调tensor维度

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)  # 对差逐元素平方
            push = torch.exp(-diff)  # 转换成指数形式
            push = torch.sum(push) - num_tags  # 归一化项
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unkown ae loss type')

        if self.o_pull_weight != 0.0:
            return push/((num_tags - 1) * num_tags) * 0.5, \
                (o_pull/(num_tags) * self.o_pull_weight + l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)
        
        if self.o_pull_weight == 0.0:
            return push/((num_tags - 1) * num_tags) * 0.5, \
                (l_pull/(num_tags) * self.l_pull_weight + h_pull/(num_tags) * self.h_pull_weight)/len(joints_per_person)
        

    def forward(self, tags, joints):
        """
        accumulate the tag loss for each image in the batch
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


@LOSSES.register_module()
class MultiLossFactory(nn.Module):
    """Loss for bottom-up models.

    Args:
        num_joints (int): Number of keypoints.
        num_stages (int): Number of stages.
        ae_loss_type (str): Type of ae loss.
        with_ae_loss (list[bool]): Use ae loss or not in multi-heatmap.
        push_loss_factor (list[float]):
            Parameter of push loss in multi-heatmap.
        pull_loss_factor (list[float]):
            Parameter of pull loss in multi-heatmap.
        with_heatmap_loss (list[bool]):
            Use heatmap loss or not in multi-heatmap.
        heatmaps_loss_factor (list[float]):
            Parameter of heatmap loss in multi-heatmap.
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self,
                 num_joints,
                 num_stages,
                 ae_loss_type,
                 with_ae_loss,
                 push_loss_factor,
                 pull_loss_factor,
                 with_heatmaps_loss,
                 heatmaps_loss_factor,
                 supervise_empty=True,
                 focal_factor=[0.01, 0.1, 0.02],
                 use_focal_loss=False,
                 use_group_loss=False,
                 use_group_focal_loss=False,
                 use_group_focal_lossv2=False,
                 o_pull_weight=0.0,
                 l_pull_weight=1.0,
                 h_pull_weight=1.0,
                 group_type='composition_coco',
                 labelloss_weight=1.0,
                 use_labelloss=False):
        super().__init__()

        assert isinstance(with_heatmaps_loss, (list, tuple)), \
            'with_heatmaps_loss should be a list or tuple'
        assert isinstance(heatmaps_loss_factor, (list, tuple)), \
            'heatmaps_loss_factor should be a list or tuple'
        assert isinstance(with_ae_loss, (list, tuple)), \
            'with_ae_loss should be a list or tuple'
        assert isinstance(push_loss_factor, (list, tuple)), \
            'push_loss_factor should be a list or tuple'
        assert isinstance(pull_loss_factor, (list, tuple)), \
            'pull_loss_factor should be a list or tuple'

        self.num_joints = num_joints
        self.num_stages = num_stages
        self.ae_loss_type = ae_loss_type
        self.with_ae_loss = with_ae_loss
        self.push_loss_factor = push_loss_factor
        self.pull_loss_factor = pull_loss_factor
        self.with_heatmaps_loss = with_heatmaps_loss
        self.heatmaps_loss_factor = heatmaps_loss_factor

        self.o_pull_weight = o_pull_weight
        self.l_pull_weight = l_pull_weight
        self.h_pull_weight = h_pull_weight
        self.group_type = group_type

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    FocalL2Loss(focal_factor[0], focal_factor[1], focal_factor[2]) 
                    if use_focal_loss else HeatmapLoss(supervise_empty)
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in self.with_heatmaps_loss
                ]
            )
        
        if use_group_loss:
            self.ae_loss = \
                nn.ModuleList(
                    [
                        AEgroupLoss(self.ae_loss_type, self.o_pull_weight, self.l_pull_weight, self.h_pull_weight, self.group_type) if with_ae_loss else None
                        for with_ae_loss in self.with_ae_loss
                    ]
                )
        else:
            if use_group_focal_loss:
                if use_group_focal_lossv2:
                    self.ae_loss = \
                        nn.ModuleList(
                            [
                                AEgroupfocalLossv2(self.ae_loss_type, self.o_pull_weight, self.l_pull_weight, self.h_pull_weight, self.group_type) if with_ae_loss else None
                                for with_ae_loss in self.with_ae_loss
                            ]
                        )
                else:
                    self.ae_loss = \
                        nn.ModuleList(
                            [
                                AEgroupfocalLoss(self.ae_loss_type) if with_ae_loss else None
                                for with_ae_loss in self.with_ae_loss
                            ]
                        )
            else:
                self.ae_loss = \
                    nn.ModuleList(
                        [
                            AELoss(self.ae_loss_type) if with_ae_loss else None
                            for with_ae_loss in self.with_ae_loss
                        ]
                    )

    def forward(self, outputs, heatmaps, masks, joints):
        """Forward function to calculate losses.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K
            - output_channel: C C=2K if use ae loss else K

        Args:
            outputs (list(torch.Tensor[N,C,H,W])): outputs of stages.
            heatmaps (list(torch.Tensor[N,K,H,W])): target of heatmaps.
            masks (list(torch.Tensor[N,H,W])): masks of heatmaps.
            joints (list(torch.Tensor[N,M,K,2])): joints of ae loss.
        """
        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred,
                                                        heatmaps[idx],
                                                        masks[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]

                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.ae_loss[idx]:
                tags_pred = outputs[idx][:, offset_feat:]
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                push_loss, pull_loss = self.ae_loss[idx](tags_pred,
                                                         joints[idx])
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses
