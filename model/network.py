import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm
from timm.models.vision_transformer import VisionTransformer
from torchvision import models
from torch.autograd import Variable
import math
# import torch.nn.utils.weight_norm as weightNorm
import torch.nn.utils.parametrizations as param
from collections import OrderedDict
import os.path as osp


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn,
            "vgg19bn": models.vgg19_bn}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d,
            "resnext101": models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        print(f"Initializing {res_name} model...")
        model_resnet = res_dict[res_name](weights='IMAGENET1K_V1')
        print(f"Model {res_name} initialized successfully with pre-trained weights.")
        # print(model_resnet)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResBase_bottleneck(nn.Module):
    def __init__(self, res_name):
        super(ResBase_bottleneck, self).__init__()
        model_resnet = res_dict[res_name](weights='IMAGENET1K_V1')
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

        self.bn_ = nn.BatchNorm1d(256, affine=True)
        self.relu_ = nn.ReLU(inplace=True)
        self.dropout_ = nn.Dropout(p=0.5)
        self.bottleneck_ = nn.Linear(self.in_features, 256)
        self.bottleneck_.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck_(x)
        x = self.bn_(x)
        return x


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori", drop_rate=0.5, normalized=False):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type
        self.normalized = normalized

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        if self.type == 'bn_drop':
            x = self.bn(x)
            x = self.dropout(x)
        if self.normalized:
            x = F.normalize(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn", bias=True):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            # self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num, bias=bias), name="weight")
            self.fc = param.weight_norm(nn.Linear(bottleneck_dim, class_num, bias=bias), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class fc_(nn.Module):
    def __int__(self):
        super(fc_, self).__int__()


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x


class rotate_classifier(nn.Module):
    def __init__(self, feature_dim, class_num, bottleneck_dim=256):
        super(rotate_classifier, self).__init__()
        self.fc0 = nn.Linear(feature_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(weights=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y


import torch.nn.functional as F
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class Predictor(nn.Module):
    def __init__(self, classifier, num_class=64, inc=4096, temp=0.05, type='wn'):
        super(Predictor, self).__init__()
        # self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        self.classifier = classifier
        # if type == 'wn':
        #     self.fc = weightNorm(nn.Linear(inc, num_class), name="weight")
        #     self.fc.apply(init_weights)
        # else:
        #     self.fc = nn.Linear(inc, num_class)
        #     self.fc.apply(init_weights)

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.classifier(x) / self.temp
        return x_out


class ViTBase(nn.Module):
    def __init__(self, vit_name, pretrained=True, freeze=False, **kwargs):
        self.KNOWN_MODELS = {
            'vit-B16': 'vit_base_patch16_224_in21k',
            'vit-B32': 'vit_base_patch32_224_in21k',
            'vit-L16': 'vit_large_patch16_224_in21k',
            'vit-L32': 'vit_large_patch32_224_in21k',
            'vit-H14': 'vit_huge_patch14_224_in21k',
            'deit-B16': 'deit_base_patch16_224'
        }

        self.FEAT_DIM = {
            'vit-B16': 768,
            'vit-B32': 768,
            'vit-L16': 1024,
            'vit-L32': 1024,
            'vit-H14': 1280,
            'deit-B16': 768  # DeiT-Base 的输出维度也是 768
        }

        super().__init__()
        self.vit_backbone = timm.create_model(
            self.KNOWN_MODELS[vit_name],
            pretrained=pretrained,
            num_classes=0,
            **kwargs
        )
        self.in_features = self.FEAT_DIM[vit_name]

        if freeze:
            for param in self.vit_backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.vit_backbone(x)



class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.args = args
        if args.net[0:3] == 'res':
            self.netF = ResBase(res_name=args.net).cuda()
        elif args.net[0:3] == 'vgg':
            self.netF = VGGBase(vgg_name=args.net).cuda()
        elif args.net.startswith(('vit', 'deit')):
            self.netF = ViTBase(vit_name=args.net).cuda()

        self.netB = feat_bootleneck(type=args.classifier, feature_dim=self.netF.in_features,
                                    bottleneck_dim=args.bottleneck).cuda()

    def forward(self, x):
        x = self.netF(x)
        x = self.netB(x)
        return x

    def load_model(self):
        if self.args.net.startswith(('vit', 'deit')):
            prefix = 'vit_'
        else:
            prefix = ''
        modelpath = osp.join(self.args.output_dir_src, f'{prefix}source_F.pt')
        self.netF.load_state_dict(torch.load(modelpath))
        modelpath = osp.join(self.args.output_dir_src, f'{prefix}source_B.pt')
        self.netB.load_state_dict(torch.load(modelpath))

