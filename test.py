import csv

with open("./others/counts.csv", "r+", encoding="gbk") as rf:
    for line in rf:
        print(line)
# import torch
# from torch_cluster import fps
#
# pos = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.], [-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
# bat = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
# # ratio = torch.tensor([0.5, 0.5], dtype=torch.float)
# ratio = 0.5
# index = fps(pos, bat, ratio=ratio, random_start=False)
# print(index)

# a = torch.arange(4).view(1, 4)
# b = torch.ones((3, )).view(3, 1)
# print(a * b)
# print((a * b).view(-1))