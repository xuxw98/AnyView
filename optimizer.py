# Copyright (c) Facebook, Inc. and its affiliates.
import torch


def build_optimizer(args, model):

    params_with_decay = []
    params_with_decay_img = []
    params_without_decay = []
    params_without_decay_img = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue
        if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
            if "im_backbone" not in name:
                params_without_decay.append(param)
            else:
                params_without_decay_img.append(param)
        else:
            if "im_backbone" not in name:
                params_with_decay.append(param)
            else:
                params_with_decay_img.append(param)

    if args.filter_biases_wd:
        param_groups = [
            {"params": params_without_decay, "weight_decay": 0.0},
            {"params": params_with_decay, "weight_decay": args.weight_decay},
            {"params": params_without_decay_img, "weight_decay": 0.0, "lr": args.image_lr},
            {"params": params_with_decay_img, "weight_decay": args.weight_decay, "lr": args.image_lr},
        ]
    else:
        param_groups = [
            {"params": params_with_decay, "weight_decay": args.weight_decay},
            {"params": params_with_decay_img, "weight_decay": args.weight_decay, "lr": args.image_lr},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.base_lr)
    return optimizer
