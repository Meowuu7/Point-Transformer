import torch
import torch.nn as nn
from model.point_transformer_net import PointTransformerObjClassificationNet
from datasets.modelnet_dataset_torch import ModelNetDataset
from torch.nn import functional as F
from tqdm import tqdm
import os
from torch.utils import data
import horovod.torch as hvd

os.environ["CUDA_VISIBLE_DEVICES"] = "3,6"
# CUDA_VISIBLE_DIVICES="3,6"

class TrainerClassification(nn.Module):
    def __init__(self, dataset_root, num_points=1024, batch_size=32, num_epochs=200, cuda=None, dataparallel=False):
        super(TrainerClassification, self).__init__()

        hvd.init()
        print(hvd.local_rank(), hvd.size(), hvd.rank())
        torch.cuda.set_device(hvd.local_rank())
        # n_layers: int, feat_dims: list, n_samples: list, n_class: int, in_feat_dim: int
        n_layers = 5
        self.num_epochs = num_epochs
        if cuda is not None:
            self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.cuda = cuda

        self.epoch_checkpoints = {120: 0.1, 160: 0.01}

        self.lr_mult_ratio = 0.9
        self.init_lr = 0.05
        self.n_points = num_points
        feat_dims = [32, 64, 128, 256, 512]
        n_samples = [self.n_points // 4, self.n_points // 16, self.n_points // 64, self.n_points // 256]
        n_class = 40
        in_feat_dim = 3
        # define model
        self.num_gpu = 1
        self.model = PointTransformerObjClassificationNet(
            n_layers=n_layers,
            feat_dims=feat_dims,
            n_samples=n_samples,
            n_class=n_class,
            in_feat_dim=in_feat_dim
        ) # .to(self.device)
        self.dataparallel = dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1
        if not self.dataparallel:
            self.model = self.model.to(self.device)
        else:
            # print("number of cuda = ", torch.cuda.device_count())
            # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 2, 3, 4, 5, 6])
            self.model = self.model.cuda()
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            # self.num_gpu = 6
            # self.init_lr *= self.num_gpu

        self.dataset_root = dataset_root
        self.train_set = ModelNetDataset(dataset_root,
                                         split='train',
                                         batch_size=batch_size,
                                         normalize=True,
                                         shuffle=True)
        self.test_set = ModelNetDataset(dataset_root,
                                        split='test',
                                        batch_size=batch_size,
                                        normalize=True,
                                        shuffle=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.init_lr,
                                         momentum=0.9,
                                         weight_decay=0.0001)
        if self.dataparallel:
            self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters())
        self.train_distributed_sampler = data.distributed.DistributedSampler(
                        self.train_set, num_replicas=hvd.size(), rank=hvd.rank())
        self.train_loader = data.DataLoader(self.train_set, batch_size=batch_size,
                                            sampler=self.train_distributed_sampler)
        self.test_distributed_sampler = data.distributed.DistributedSampler(
            self.test_set, num_replicas=hvd.size(), rank=hvd.rank()
        )
        self.test_loader = data.DataLoader(self.test_set, batch_size=batch_size,
                                           sampler=self.test_distributed_sampler)


    def adjust_learning_rate(self, epoch):
        if epoch in self.epoch_checkpoints:
            new_lr = self.init_lr * self.epoch_checkpoints[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def calculate_acc(self, pred: torch.LongTensor, gt: torch.LongTensor):
        # cn = (pred == gt).sum() # gt is not entached emm...
        return (torch.sum(torch.max(pred.detach(), dim=1)[1] == gt).cpu().item())

    def _train_one_epoch(self, epoch):
        self.model.train()
        if epoch in self.epoch_checkpoints:
            self.adjust_learning_rate(epoch)

        tot_acc = 0
        tot_num = 0
        loss_list = []
        loss_nn = []
        # step = 0
        # train_bar = tqdm(self.train_set)
        train_bar = tqdm(self.train_loader)
        for batch_data, batch_label in train_bar:
            # while self.train_set.has_next_batch():
            #     batch_data, batch_label = self.train_set.next_batch()
            # batch_x = torch.from_numpy(batch_data).float() # .to(self.device)
            # batch_pos = torch.from_numpy(batch_data).long() # .to(self.device)
            # batch_label = torch.from_numpy(batch_label).long() # .to(self.device)
            batch_x = batch_data.float()
            batch_pos = batch_data.long()
            batch_label = batch_label.long()
            if not self.dataparallel:
                batch_x = batch_x.to(self.device)
                batch_pos = batch_pos.to(self.device)
                batch_label = batch_label.to(self.device)
            else:
                batch_x = batch_x.cuda()
                batch_pos = batch_pos.cuda()
                batch_label = batch_label.cuda()

            bz = batch_x.size(0)
            logits = self.model(x=batch_x, pos=batch_pos)
            loss = F.nll_loss(input=F.log_softmax(logits, dim=-1), target=batch_label)
            # loss /= self.num_gpu
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred_labels = self.calculate_acc(logits, batch_label)
            tot_acc += pred_labels
            tot_num += logits.size(0)
            loss_list += [loss.detach().cpu().item() * bz]
            loss_nn.append(bz)
            # print("Epoch: %d, loss = %.4f, acc = %.4f." % (epoch,
            #                                            float(sum(loss_list) / sum(loss_nn)),
            #                                            float(tot_acc / tot_num)))
            # jiangzia... *100...
            train_bar.set_description(
                'Train Epoch: [{}/{}] Loss:{:.3f} Acc@1:{:.2f}%'.format(epoch, 200,
                                                                      float(sum(loss_list) / sum(loss_nn)),
                                                                      float(tot_acc / tot_num)))
        self.train_set.reset()

    def _test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            tot_acc = 0
            tot_num = 0
            # test_bar = tqdm(self.test_set)
            test_bar = tqdm(self.test_loader)
            for batch_data, batch_label in test_bar:
                # while self.test_set.has_next_batch():
                #     batch_data, batch_label = self.test_set.next_batch()
                # batch_x = torch.from_numpy(batch_data).float()  # .to(self.device)
                # batch_pos = torch.from_numpy(batch_data).long()  # .to(self.device)
                # batch_label = torch.from_numpy(batch_label).long()  # .to(self.device)
                batch_x = batch_data.float()
                batch_pos = batch_data.long()
                batch_label = batch_label.long()
                if not self.dataparallel:
                    batch_x = batch_x.to(self.device)
                    batch_pos = batch_pos.to(self.device)
                    batch_label = batch_label.to(self.device)
                else:
                    batch_x = batch_x.cuda()
                    batch_pos = batch_pos.cuda()
                    batch_label = batch_label.cuda()
                logits = self.model(x=batch_x, pos=batch_pos)

                pred_labels = self.calculate_acc(logits, batch_label)
                tot_acc += pred_labels
                tot_num += logits.size(0)
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch + 1, 200,
                                                                                float(tot_acc / tot_num)))
            # print("Epoch: %d, test_acc = %.4f" % (epoch, float(tot_acc / tot_num)))
            self.test_set.reset()

    def train_all(self):
        print("Start training.")
        for epoch in range(self.num_epochs):
            self._train_one_epoch(epoch + 1)
            if epoch % 10 == 0:
                self._test(epoch)

