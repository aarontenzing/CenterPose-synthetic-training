# Python program to move a tensor from CPU to GPU
# import torch library
import torch

# create a tensor
x = torch.tensor([1.0,2.0,3.0,4.0])
print("Tensor:", x)

# check tensor device (cpu/cuda)
print("Tensor device:", x.device)

# Move tensor from CPU to GPU
# check CUDA GPU is available or not
print("CUDA GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
   x = x.to("cuda:0")
   # or x=x.to("cuda")
print(x)

# now check the tensor device
print("Tensor device:", x.device)


for i in range(0,torch.cuda.device_count()):
   print(torch.cuda.get_device_name(device=i))