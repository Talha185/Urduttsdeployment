import torch
print(torch.cuda.is_available())         # should be True
print(torch.cuda.get_device_name(0))     # should print your GPU name
# $env:PYTHONPATH += ";D:\Rippleberry\ttsbackend\vits"
# python ../main.py
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))