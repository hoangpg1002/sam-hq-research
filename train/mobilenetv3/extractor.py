import torch
import sys
sys.path.append("D:\StableDiffusion\sam-hq\mobilenetv3")
from mobilenet import mobilenetv3_large
def extract_feature(inputs):
    net_large = mobilenetv3_large().to(device="cuda")
    #net_large.load_state_dict(torch.load('D:\StableDiffusion\sam-hq\mobilenetv3\pretrained\mobilenetv3-large-1cd25616.pth'))
    for param in net_large.parameters():
        param.requires_grad = False
    with torch.no_grad():
        feature_extractor=net_large.features

    outputs_large = feature_extractor(inputs)
    return outputs_large
x= torch.randn(1,3,1024,1024).to(device="cuda")
out=extract_feature(x)
print(out.shape)