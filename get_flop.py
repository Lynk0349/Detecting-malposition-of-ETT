import torch

from model.fcos import FCOSDetector
from thop import profile


model=FCOSDetector(mode="inference")
model = model.to("cuda:0")
input = torch.randn(1,3,224,224).to("cuda:0")
flop, para = profile(model, inputs=(input,), verbose=False)
print("GFlop: %.2f" % (flop/1e9),"\nParameters: %.2fM" % (para/1e6))