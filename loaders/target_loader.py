import torch.utils.data

from loaders.data_list import ImageList, ImageList_idx, Two_ImageList_idx
from utils.tools import *
from torch.utils.data import DataLoader


def data_load(args, distributed=False):
    strong_aug = {
                  'mocov2': mocov2(),
                  }
    # prepare data
    dsets = {}
    dset_loaders = {}
    samplers = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i  # map

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')  # ['img[i]_path','class']
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    # only be forwarded when da=='oda'
                    # class that not exist in both source and target domains are concluded to one class.
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train(), append_root=args.append_root)
    if distributed:
        samplers['target'] = torch.utils.data.distributed.DistributedSampler(dsets["target"], shuffle=True)
    else:
        samplers['target'] = None
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, num_workers=args.worker,
                                        shuffle=(distributed is False),
                                        drop_last=False, sampler=samplers['target'], pin_memory=True)

    dsets["test"] = ImageList_idx(txt_test, transform=image_test(), append_root=args.append_root)
    # no shuffle
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False, sampler=None, pin_memory=True)
    if distributed:
        samplers['test'] = torch.utils.data.DistributedSampler(dsets['test'], shuffle=False)
        # no shuffle
        dset_loaders["d_test"] = DataLoader(dsets["test"], batch_size=train_bs, num_workers=args.worker,
                                            drop_last=False, sampler=samplers['test'], pin_memory=True)

    dsets["two_train"] = Two_ImageList_idx(txt_tar, transform1=image_train(), transform2=strong_aug[args.aug], append_root=args.append_root)
    if distributed:
        samplers['two_train'] = torch.utils.data.distributed.DistributedSampler(dsets["two_train"], shuffle=True)
    else:
        samplers['two_train'] = None
    dset_loaders["two_train"] = DataLoader(dsets["two_train"], batch_size=train_bs, num_workers=args.worker,
                                           shuffle=True,
                                           drop_last=False, sampler=samplers['two_train'], pin_memory=True)
    # dsets["two_train_test"] = ImageList_multi_transform(txt_tar, transform=[image_train(), strong_aug[args.aug], image_test()])
    # dset_loaders["two_train_test"] = DataLoader(dsets["two_train_test"], batch_size=train_bs, num_workers=args.worker,
    #                                             shuffle=True, drop_last=False, sampler=None, pin_memory=True)
    # dset_loaders['queue_two_train'] = DataLoader(dsets["two_train"], batch_size=train_bs, num_workers=)
    if distributed:
        return dset_loaders, samplers
    else:
        return dset_loaders, dsets
