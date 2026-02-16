import torch

print("Torch version:", torch.__version__)

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

result = x + y

print("Tensor addition result:")
print(result)
