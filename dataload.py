from torch.utils.data import DataLoader
from torchvision import transforms
from train.utils.dataloader_rs import create_dataloaders
from train.utils.dataloader_rs import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
import matplotlib.pyplot as plt
   ### --------------- Configuring the Train and Valid datasets ---------------
dataset_dis = {"name": "DIS5K-TR",
                "im_dir": "./data/DIS5K/DIS-TR/im",
                "gt_dir": "./data/DIS5K/DIS-TR/gt",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_thin = {"name": "ThinObject5k-TR",
                "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_fss = {"name": "FSS",
                "im_dir": "./data/cascade_psp/fss_all",
                "gt_dir": "./data/cascade_psp/fss_all",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_duts = {"name": "DUTS-TR",
                "im_dir": "./data/cascade_psp/DUTS-TR",
                "gt_dir": "./data/cascade_psp/DUTS-TR",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_duts_te = {"name": "DUTS-TE",
                "im_dir": "./data/cascade_psp/DUTS-TE",
                "gt_dir": "./data/cascade_psp/DUTS-TE",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_ecssd = {"name": "ECSSD",
                "im_dir": "./data/cascade_psp/ecssd",
                "gt_dir": "./data/cascade_psp/ecssd",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_msra = {"name": "MSRA10K",
                "im_dir": "./data/cascade_psp/MSRA_10K",
                "gt_dir": "./data/cascade_psp/MSRA_10K",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

# valid set
dataset_coift_val = {"name": "COIFT",
                "im_dir": "./data/thin_object_detection/COIFT/images",
                "gt_dir": "./data/thin_object_detection/COIFT/masks",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_hrsod_val = {"name": "HRSOD",
                "im_dir": "./data/thin_object_detection/HRSOD/images",
                "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_thin_val = {"name": "ThinObject5k-TE",
                "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
                "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

dataset_dis_val = {"name": "DIS5K-VD",
                "im_dir": "./data/DIS5K/DIS-VD/im",
                "gt_dir": "./data/DIS5K/DIS-VD/gt",
                "im_ext": ".jpg",
                "gt_ext": ".png"}

train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 

print("--- create training dataloader ---")
train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                my_transforms = [
                                                            RandomHFlip(),
                                                            LargeScaleJitter()
                                                            ],
                                                batch_size = 8,
                                                training = True)
# print(len(train_dataloaders), " train dataloaders created")




