import torch

# print(torch.cuda.is_available())

a = torch.randint(0,100,[2,2,2])
b = torch.randint(0,100,[2,2,1])
c = 1000
print(a)
print(a*c)