"""Download official ImageNet weights; no product-specific training is claimed."""
from pathlib import Path
import torch
from torchvision.models import MobileNet_V3_Small_Weights
root=Path(__file__).resolve().parents[1]
root.joinpath('models').mkdir(exist_ok=True)
weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
state=weights.get_state_dict(progress=True,check_hash=True)
torch.save(state,root/'models/mobilenet_v3_small.pth')
print('Saved official pretrained MobileNet V3 Small weights.')
