import torch
a = torch.Tensor([[1,2,3, 4],
                  [5,6,0, 7],
                  [4,5,6, 4]])
v, i = torch.max(a.reshape(-1), dim=0)
x = i / 4
y = i - x * 4
print(x, y)

