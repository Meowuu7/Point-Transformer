import os
import os.path
import json
import numpy as np
import sys
# from datasets.modelnet_dataset import pc_normalize
from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def preprocess_dataset(root, npoints=1024, split='train', normalize=True, normal_channel=False,
                 modelnet10=False, debug=True, rng=None):
    if modelnet10:
        catfile = os.path.join(root, 'modelnet10_shape_names.txt')
    else:
        catfile = os.path.join(root, 'modelnet40_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))
    shape_ids = {}
    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_train.txt'))]
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_test.txt'))]
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
    datapath = [(shape_names[i], os.path.join(root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                in range(len(shape_ids[split]))]

    n = len(datapath)
    print("split = ", split, "; n = ", n)
    idx_to_point_set_cls = {}
    l, r = 0, n
    if rng is not None:
        l, r = rng
    r = r if r <= n else n
    for index in range(l, r):
        if index % 50 == 0:
            print(index)
        if debug and (index - l) > 100:
            break
        fn = datapath[index]
        cls = classes[datapath[index][0]]
        cls = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        # Take the first npoints
        point_set = point_set[0: npoints, :]
        if normalize:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # if not normal_channel:
        #     point_set = point_set[:, 0:3]
        idx_to_point_set_cls[index] = (point_set, cls)
    np.save(os.path.join(root, "modelnet40_{}_npoints_{}_fea_6_l_{:d}.npy".format(split, str(npoints), l)), idx_to_point_set_cls)
    print("saved! l = ", l)

def preprocess_dataset_all(root, npoints=1024, split='train', normalize=True, normal_channel=False,
                 modelnet10=False, debug=True):
    n_tot = 10000 if split == 'train' else 3000
    n_cpu = 25 if split == 'train' else 20
    per = n_tot // n_cpu
    # per = 400
    rngs = [(i * per, (i + 1) * per) for i in range(n_cpu)]
    po = Pool(n_cpu)
    for rng in rngs:
        po.apply_async(preprocess_dataset, (root, npoints, split, normalize, normal_channel, modelnet10, debug, rng))
    po.close()
    po.join()

def remove_files(root, prefix):
    for root, dirs, files in os.walk(root, topdown=False):
        for name in files:
            if name.startswith(prefix):
                os.remove(os.path.join(root, name))

def check_saved(root, split, npoints=1024):
    shape_ids = {}
    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_train.txt'))]
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_test.txt'))]
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
    datapath = [(shape_names[i], os.path.join(root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                in range(len(shape_ids[split]))]

    n = len(datapath)
    print("split = ", split, "; n = ", n)
    n_tot = 10000 if split == 'train' else 3000
    n_cpu = 25 if split == 'train' else 20
    per = n_tot // n_cpu
    rngs = [(i * per, (i + 1) * per) for i in range(n_cpu)]
    # for rng in rngs:
    #     print(rng)
    #     idx_to_point_set_cls = np.load(os.path.join(root, "modelnet40_{}_npoints_{}_fea_6_l_{:d}.npy".format(split,
    #                                                                                                    str(npoints),
    #                                                                                                    rng[0]))).item()
    #     print(type(idx_to_point_set_cls))
    #     print(len(idx_to_point_set_cls))
    # for j in idx_to_point_set_cls:
    #     print(idx_to_point_set_cls[j])
    idx_to_point_set_cls = np.load(os.path.join(root, "modelnet40_{}_npoints_{}_fea_6.npy".format(split,
                                                                                            str(npoints)))).item()
    print(len(idx_to_point_set_cls))

def merge_saved_data(root, split, npoints=1024):
    n_tot = 10000 if split == 'train' else 3000
    n_cpu = 25 if split == 'train' else 20
    per = n_tot // n_cpu
    rngs = [(i * per, (i + 1) * per) for i in range(n_cpu)]
    res = {}
    for rng in rngs:
        print(rng)
        idx_to_point_set_cls = np.load(os.path.join(root, "modelnet40_{}_npoints_{}_fea_6_l_{:d}.npy".format(split,
                                                                                                       str(npoints),
                                                                                                       rng[0]))).item()
        for idx in idx_to_point_set_cls:
            assert idx not in res
            res[idx] = idx_to_point_set_cls[idx]
    np.save(os.path.join(root, "modelnet40_{}_npoints_{}_fea_6.npy".format(split, str(npoints))), res)

if __name__=='__main__':
    # preprocess_dataset("./data/modelnet40_normal_resampled",
    #                    npoints=1024, normalize=True, modelnet10=False)
    # check_saved("./data/modelnet40_normal_resampled", "train")
    # preprocess_dataset_all("./data/modelnet40_normal_resampled", split='test',
    #                         npoints=1024, normalize=True, modelnet10=False, debug=False)
    # remove_files("./data/modelnet40_normal_resampled", "modelnet40_train_npoints_1024_l_")
    check_saved("./data/modelnet40_normal_resampled", "test")
    # merge_saved_data("./data/modelnet40_normal_resampled", "test")
