import csv

# with open("./others/counts.csv", "r+", encoding="gbk") as rf:
#     for line in rf:
#         print(line)
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

# fn = "./datasets/indoor3d_sem_seg_hdf5_data/room_filelist.txt"
# res = {}
# with open(fn, "r") as rf:
#     for line in rf:
#         ss = line.strip()
#         if ss in res:
#             res[ss] += 1
#         else:
#             res[ss] = 1
#
# print(len(res))
fn = "./others/point_1139_view_2_domain_rgb.png"
from PIL import Image
im = Image.open(fn)
pix = im.load()
width = im.size[0]
height = im.size[1]
print(width, height)
pixval = {}
for x in range(width):
    for y in range(height):
        # print(pix[x, y])
        tmpval = pix[x, y]
        if tmpval not in pixval:
            pixval[tmpval] = 1
        else:
            pixval[tmpval] += 1
        # r, g, b = pix[x, y]

nn = 0
for pv in pixval:
    print(pv, pixval[pv])
    nn += pixval[pv]
print(nn, width * height) # 256 * 256 = 65536!
