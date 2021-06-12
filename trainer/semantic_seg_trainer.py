import torch
import torch.nn as nn
# from model.point_transformer_net import PointTransformerObjClassificationNet
from model.point_transformer_seg_net import PointTransformerSemanticSegNet
# from datasets.modelnet_dataset_torch import ModelNetDataset
from datasets.Indoor3DSeg_dataset import Indoor3DSemSeg
from torch.nn import functional as F
from tqdm import tqdm
import os
from torch.utils import data
from . import provider


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"

class TrainerSegmentation(nn.Module):
    def __init__(self, dataset_root, num_points=1024, batch_size=32, num_epochs=200, cuda=None, dataparallel=False,
                 fea_num=3, use_sgd=False, more_aug=False, weight_decay_sgd=5e-4, lr_scheduler="step",
                 resume="", dp_ratio=0.5, attn_mult=2, args=None):
        super(TrainerSegmentation, self).__init__()
        # n_layers: int, feat_dims: list, n_samples: list, n_class: int, in_feat_dim: int
        n_layers = 5
        self.num_epochs = num_epochs
        if cuda is not None:
            self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.cuda = cuda
        self.fea_num = fea_num
        self.more_aug = more_aug
        self.lr_scheduler = lr_scheduler
        self.dp_ratio = dp_ratio
        self.attn_mult = attn_mult

        self.epoch_checkpoints = {120: 0.1, 160: 0.01}

        self.lr_mult_ratio = 0.9
        # self.init_lr = 0.005
        self.init_lr = 0.001
        # self.init_lr = 0.0005
        self.weight_decay = 1e-4
        self.weight_decay_sgd = weight_decay_sgd
        self.n_points = num_points
        feat_dims = [32, 64, 128, 256, 512]
        n_samples = [self.n_points // 4, self.n_points // 16, self.n_points // 64, self.n_points // 256]
        n_class = 40
        in_feat_dim = 3
        in_feat_dim = self.fea_num
        # define model
        self.num_gpu = 1
        self.model = PointTransformerSemanticSegNet(
            n_layers=n_layers,
            feat_dims=feat_dims,
            n_samples=n_samples,
            n_class=n_class,
            in_feat_dim=in_feat_dim,
            dp_ratio=dp_ratio,
            attn_mult=attn_mult,
            args=args
        ) # .to(self.device)

        if len(resume) != 0:
            self.model.load_state_dict(torch.load(resume, map_location='cpu'))
        self.model.to(self.device)
        self.dataparallel = dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1
        # if not self.dataparallel:
        #     print(self.device)
        #     self.model = self.model.to(self.device)
        # TODO: the influence of lr on model's performance and whether easy to train
        # else: 3 -- 0.01; 5 -- 0.03; 1 -- 0.05 4 -- 0.005 p6 -- 0.005 + pt_transfomer_projection
        # parallel train --- test to 0.78
        # 1 -- 0.05 -- train acc from 0.72 -> 0.80.. --- we cannot use 0.05 anymore its too large.. --- test to 0.80 at 121
        # 5 -- 0.03 -- test@121 = 0.82 similar increase for test acc also observed on this learning rate setting...
        #     print("number of cuda = ", torch.cuda.device_count())
        #     self.model = torch.nn.DataParallel(self.model, device_ids=[0, 2, 3, 4, 5, 6])
        #     self.model = self.model.cuda()
        #     self.num_gpu = 6
        #     self.init_lr *= self.num_gpu

        # if cuda is not None:
        #     if torch.cuda.is_available():
        #         if torch.cuda.device_count() > 1:
        #             print("number of aval cuda = ", torch.cuda.device_count())
        #             self.model = torch.nn.DataParallel(self.model)

        # root, batch_size=32, npoints=1024, split='train', normalize=True, normal_channel=False,
        #                  modelnet10=False, cache_size=15000, shuffle=None
        self.dataset_root = dataset_root
        self.train_set = Indoor3DSemSeg(num_points=num_points, train=True, download=True, data_precent=1.0)
        self.test_set = Indoor3DSemSeg(num_points=num_points, train=False, download=True, data_precent=1.0)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=self.init_lr,
        #                                  momentum=0.9,
        #                                  weight_decay=0.0001)
        self.use_sgd = use_sgd
        if not self.use_sgd:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.init_lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.3)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.init_lr,
                                             momentum=0.9,
                                             weight_decay=self.weight_decay_sgd,
                                             )
            if self.lr_scheduler == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            T_max=50,
                                                                            eta_min=0.00001,
                                                                            )
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.3)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.3)
        self.train_loader = data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(self.test_set, batch_size=batch_size, shuffle=True)

        self.model_dir = "semantic_seg_no_ext_no_drp_init_lr_{}_bsz_{}_drop_lr_120_{}_160_{}_Adam_{}_50_lr_schedule_projection_aug_more_{}_nfeat_{}_weight_decay_{}_with_dropout_{}_more_bn_attn_mult_{}_use_abs_pos_{}_with_normal_{}_resume_{}".format(str(self.init_lr),
                                                                          str(batch_size),
                                                                          str(self.epoch_checkpoints[120]),
                                                                          str(self.epoch_checkpoints[160]),
                                                                          str(not self.use_sgd),
                                                                          str(self.more_aug),
                                                                          str(self.fea_num),
                                                                          str(self.weight_decay_sgd if use_sgd else self.weight_decay),
                                                                          str(self.dp_ratio),
                                                                          str(self.attn_mult),
                                                                          str(args.use_abs_pos),
                                                                          str(args.with_normal),
                                                                          str(True if len(resume) > 0 else False))
        # TODO: different initial learning rate? try 0.0005 as the initial learning rate with different scheduler
        #       For regs: (1) I don't think more data augmentation is a good idea. The network architecture is not
        #       changed if we just change the input data; (2) Try add some reguralization items in the loss;
        #       For hidden dimensions and other hyperparameters: (1) over-fitting can be observed in later training
        #       process (acc for train set can increase apparently at the learning rate dropping epochs;
        # TODO: Try add some L2-reguralization and dropout for network weights; Try 0.0005 as the init_lr;
        #       try to add dropout layers between two fully-connected layers --- just limited places to add;
        # TODO: Add bn layers after each fully-connected layer and before the activation layer
        # TODO: other lr scheduler; add bn layers between each fully-connected layers;
        # TODO: bn monument????? --- Point-Transformer project set the monument to the default value (say 0.1)
        #  but implementation details in paper DGCNN set the monument to 0.9? which is correct?
        #  Moreover, bn_decay? grad_norm_clip? increase the model's capacity with dropout and bn
        #  A strange thing is that the model cannot even fit train set if the dropout ratio is set to 0.7; -- resume and retrain?
        if not os.path.exists("./cache"):
            os.mkdir("./cache")
        if not os.path.exists(os.path.join("./cache", self.model_dir)):
            os.mkdir(os.path.join("./cache", self.model_dir))
        self.model_dir = "./cache/" + self.model_dir

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
        if self.use_sgd and self.lr_scheduler != "cosine":
            if epoch in self.epoch_checkpoints:
                self.adjust_learning_rate(epoch)

        tot_acc = 0
        tot_num = 0
        loss_list = []
        loss_nn = []
        step = 0
        # train_bar = tqdm(self.train_set)
        train_bar = tqdm(self.train_loader)
        for batch_data, batch_label in train_bar:
            # while self.train_set.has_next_batch():
            #     batch_data, batch_label = self.train_set.next_batch()
            # batch_x = torch.from_numpy(batch_data).float() # .to(self.device)
            # batch_pos = torch.from_numpy(batch_data).long() # .to(self.device)
            # batch_label = torch.from_numpy(batch_label).long() # .to(self.device)

            # point data augmentation
            points = batch_data.float().data.numpy()
            # more augmentation
            # points = provider.random_point_dropout(points)
            if self.more_aug:
                points = provider.rotate_point_cloud_with_normal(points)
                points = provider.rotate_perturbation_point_cloud_with_normal(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            if self.more_aug:
                points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            # # target = target[:, 0]
            batch_x = points.float()
            batch_pos = points.float()

            # batch_x = batch_data.float()
            # batch_pos = batch_data.float()
            batch_label = batch_label.long()
            # print(batch_label.size())
            if len(batch_label.size()) > 1:
                batch_label = batch_label.view(-1)
            if not self.dataparallel:
                batch_x = batch_x.to(self.device)
                batch_pos = batch_pos.to(self.device)
                batch_label = batch_label.to(self.device)
            else:
                batch_x = batch_x.cuda()
                batch_pos = batch_pos.cuda()
                batch_label = batch_label.cuda()
            bz, N = batch_x.size(0), batch_x.size(1)
            logits = self.model(x=batch_x, pos=batch_pos)
            logits = logits.view(bz * N, -1)
            batch_label = batch_label.view(-1)
            # print(logits.size(), batch_label.size())
            loss = F.nll_loss(input=F.log_softmax(logits, dim=-1), target=batch_label) # avg for each pixel
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
                                                                      float(tot_acc / tot_num) * 100))
        with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
            wf.write("Train Epoch: {:d}, loss: {:.4f}, Acc: {:.2f}%\t".format(epoch + 1,
                                                                              float(sum(loss_list) / sum(loss_nn)),
                                                                              float(tot_acc / tot_num) * 100))
            wf.close()
        if self.lr_scheduler == "cosine" or (not self.use_sgd):
            self.scheduler.step()
        # self.train_set.reset()

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
                batch_pos = batch_data.float()
                batch_label = batch_label.long()
                if len(batch_label.size()) > 1:
                    batch_label = batch_label.view(-1)
                if not self.dataparallel:
                    batch_x = batch_x.to(self.device)
                    batch_pos = batch_pos.to(self.device)
                    batch_label = batch_label.to(self.device)
                else:
                    batch_x = batch_x.cuda()
                    batch_pos = batch_pos.cuda()
                    batch_label = batch_label.cuda()
                bz, N = batch_x.size(0), batch_x.size(1)
                logits = self.model(x=batch_x, pos=batch_pos)

                logits = logits.view(bz * N, -1)
                batch_label = batch_label.view(-1)

                pred_labels = self.calculate_acc(logits, batch_label)
                tot_acc += pred_labels
                tot_num += logits.size(0)
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch + 1, 200,
                                                                                float(tot_acc / tot_num) * 100))
            # print("Epoch: %d, test_acc = %.4f" % (epoch, float(tot_acc / tot_num)))
            with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                wf.write("Test Epoch: {:d}, Acc: {:.2f}%\n".format(epoch + 1, float(tot_acc / tot_num) * 100))
                wf.close()
            # self.test_set.reset()

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "checkpoint_current.pth".format(epoch)))

    def train_all(self):
        print("Start training.")
        for epoch in range(self.num_epochs):
            self._train_one_epoch(epoch + 1)

            self._test(epoch)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print("=== Saving model")
                self.save_model(epoch + 1)
                print("=== Saved")

