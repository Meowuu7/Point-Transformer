# from trainer import TrainerClassification
import argparse
import torch
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,6"

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_fea', type=int, default=3,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--dataparallel', action="store_true", default=False,
                    help='learning rate (default: 0.001)')
parser.add_argument('--use_sgd', action="store_true", default=False,
                    help='learning rate (default: 0.001)')
parser.add_argument('--more_aug', action="store_true", default=False,
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay_sgd', type=float, default=0.0005,
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr_scheduler', type=str, default="step",
                    help='learning rate (default: 0.001)')
parser.add_argument('--resume', type=str, default="",
                    help='learning rate (default: 0.001)')
parser.add_argument('--dp_ratio', type=float, default=0.5,
                    help='learning rate (default: 0.001)')
parser.add_argument('--attn_mult', type=int, default=2,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--use_abs_pos', action="store_true", default=False,
                    help='learning rate (default: 0.001)')
parser.add_argument('--with_normal', action="store_true", default=False,
                    help='learning rate (default: 0.001)')
parser.add_argument('--task', type=str, default="cls",
                    help='learning rate (default: 0.001)')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.dataparallel:
        from trainer.cls_trainer_hvd import TrainerClassification as trainer
    else:
        if args.task == "cls":
            from trainer.cls_trainer import TrainerClassification as trainer
        elif args.task == "sem_seg":
            from trainer.semantic_seg_trainer import TrainerSegmentation as trainer
        else:
            raise ValueError("Unrecognized task name. Expected cls or sem_seg, got %s" % args.task)

    torch.manual_seed(0)
    np.random.seed(0)
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    cudaa = args.device if torch.cuda.is_available() else None
    #  dataset_root, num_points=1024, batch_size=32, num_epochs=200, cuda=None
    # /home/lxyww7/point-transformer-pytorch/cache/init_lr_0.001_bsz_16_drop_lr_120_0.1_160_0.01_Adam_False_50_lr_schedule_projection_aug_more_False_nfeat_6_weight_decay_0.0005_with_dropout
    # --use_sgd --num_fea=6 --device=0 --batch_size=16
    # --resume="./cache/init_lr_0.001_bsz_16_drop_lr_120_0.1_160_0.01_Adam_False_50_lr_schedule_projection_aug_more_False_nfeat_6_weight_decay_0.0005_with_dropout/checkpoint_epoch_200.pth"
    # --use_sgd --num_fea=9 --device=6 --batch_size=16 --dp_ratio=0.5 --attn_mult=1 --use_abs_pos --task=sem_seg --resume="./cache/semantic_seg_no_ext_init_lr_0.001_bsz_16_drop_lr_120_0.1_160_0.01_Adam_False_50_lr_schedule_projection_aug_more_False_nfeat_9_weight_decay_0.0005_with_dropout_0.4_more_bn_attn_mult_1_use_abs_pos_True_with_normal_False_resume_False/checkpoint_epoch_40.pth"
    # --with_normal
    # --use_sgd --num_fea=6 --device=6 --batch_size=16 --dp_ratio=0.3 --attn_mult=1 --use_abs_pos --more_aug
    print("dataparallel = ", args.dataparallel)
    print("number_feature = ", args.num_fea)
    trainer = trainer(dataset_root="./data/modelnet40_normal_resampled",
                      num_points=1024,
                      batch_size=args.batch_size,
                      num_epochs=200,
                      cuda=cudaa,
                      dataparallel=args.dataparallel,
                      fea_num=args.num_fea,
                      use_sgd=args.use_sgd,
                      more_aug=args.more_aug,
                      weight_decay_sgd=args.weight_decay_sgd,
                      lr_scheduler=args.lr_scheduler,
                      resume=args.resume,
                      dp_ratio=args.dp_ratio,
                      attn_mult=args.attn_mult,
                      args=args)
    trainer.train_all()
