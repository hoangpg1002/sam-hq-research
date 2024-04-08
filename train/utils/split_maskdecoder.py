import torch
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc


# checkpoint_dict = {
#     "vit_b": "pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
#     "vit_l": "pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
#     "vit_h": "pretrained_checkpoint/sam_vit_h_maskdecoder.pth"
# }

# checkpoint_path = checkpoint_dict["vit_b"]
# model_state_dict = torch.load(checkpoint_path)
# model_state_dict1=torch.load("D:\StableDiffusion\sam-hq\sam_vit_b_maskdecoder_test.pth")
# model = MaskDecoder(num_multimask_outputs=3,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=256,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=256,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,)
# model1 = MaskDecoder(num_multimask_outputs=3,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=256,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=256,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,)
# model.load_state_dict(model_state_dict)
# model1.load_state_dict(model_state_dict1)
# tensor1 = torch.tensor([])
# tensor2 = torch.tensor([])
# # Setting requires_grad to False for all parameters
# for name, param in model.named_parameters():
#     tensor1 = torch.cat((tensor1, param.view(-1)))
# for name, param in model1.named_parameters():
#     tensor2 = torch.cat((tensor2, param.view(-1)))
# parameters_equal = torch.allclose(tensor1, tensor2)
# print(parameters_equal)

sam = sam_model_registry["vit_b"](checkpoint="train\pretrained_checkpoint\sam_vit_b_01ec64.pth")
mask_decoder_state_dict = {}
for name, param in sam.named_parameters():
    if name.startswith("mask_decoder"):
        name=name.replace("mask_decoder.","")
        mask_decoder_state_dict[name] = param
save_path = "sam_vit_b_maskdecoder_test.pth"
torch.save(mask_decoder_state_dict, save_path)