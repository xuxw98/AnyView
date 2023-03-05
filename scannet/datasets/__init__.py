# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetSVDetectionDataset, ScannetAnyViewDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "scannetSV": [ScannetSVDetectionDataset, ScannetDatasetConfig],
    "scannetAV": [ScannetAnyViewDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    if args.dataset_name == "scannet":
        dataset_dict = {
            "train": dataset_builder(
                dataset_config, 
                split_set="train", 
                root_dir=args.dataset_root_dir, 
                use_color=args.use_color,
                augment=True,
                use_depth_fusion=args.use_depth_fusion
            ),
            "test": dataset_builder(
                dataset_config, 
                split_set="val", 
                root_dir=args.dataset_root_dir, 
                use_color=args.use_color,
                augment=False,
                use_depth_fusion=args.use_depth_fusion
            ),
        }
    elif args.dataset_name == "scannetAV":
        dataset_dict = {
            "train": dataset_builder(
                dataset_config, 
                split_set="train", 
                root_dir=args.dataset_root_dir, 
                use_color=args.use_color,
                augment=True,
                use_random_cuboid=args.use_random_cuboid,
                use_view_drop=args.use_view_drop,
                num_views=args.num_views,
                continue_view=args.continue_view,
                without_trans=args.without_trans
            ),
            "test": dataset_builder(
                dataset_config, 
                split_set="val", 
                root_dir=args.dataset_root_dir, 
                use_color=args.use_color,
                augment=False,
                num_views=50,
                without_trans=args.without_trans,
                test_anyview=args.test_anyview
            ),
        }
    else:
        dataset_dict = {
            "train": dataset_builder(
                dataset_config, 
                split_set="train", 
                root_dir=args.dataset_root_dir, 
                use_color=args.use_color,
                augment=True,
                without_trans=args.without_trans
            ),
            "test": dataset_builder(
                dataset_config, 
                split_set="val", 
                root_dir=args.dataset_root_dir, 
                use_color=args.use_color,
                augment=False,
                without_trans=args.without_trans
            ),
        }
    return dataset_dict, dataset_config
    
