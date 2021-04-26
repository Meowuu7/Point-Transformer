## Pytorch Implementation of Point Transformer

- Implementation of the <a href="https://arxiv.org/abs/2012.09164">Point Transformer</a> in Pytorch. 
- Some code are borrowed from [Pointnet++](https://github.com/charlesq34/pointnet2) and [point-transformer-pytorch](https://github.com/lucidrains/point-transformer-pytorch). 

### Classification

#### Data Preparation

- Download [**ModelNet40**](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled`. 

#### Run

- Please use the following command to train and evaluate point-transformer on ModelNet40 for shape classification task:

  ```shell
  python3 main_training.py --num_feat=${num_feat} \ # number of features used for points in the shape cloud
  												 --device=${device} \ # which gpu to use in the training process
  												 --batch_size=${batch_size} \ # batch size
  												 --dp_ratio=${dropout_ratio} \ # dropout ratio
  												 --attn_mult=${attn_mult} \ # for point-transformer layer
  												 [--use_abs_pos] \ # whether to use absolute position information in point-transformer layer
  												 [--with_normal] \ # whether to use points' normal information in point-transformer layer
  												 [--more_aug] # whether to use more data augmentation in training process
  ```

#### Results

- Use SGD with initial learning rate 0.001 and decay by 0.1 in epoch 120 and 160, train for 200 epochs; batch size is set to 16; dropout lyaers are added between each two fully-connected layers with the dropout ratio set to 0.4, using absolute position information in the point-transformer layers. 

- Instance classification accuracy on ModelNet40 are as follows:

  |       | OA    |
  | ----- | ----- |
  | Paper | 93.7% |
  | Ours  | 92.6% |

- (The performance can be probably improved if the dropout ratio, initial learning rate and the learning rate schduler could be further fine-tuned.) 

### Miscellaneous

We look forward to authors of <a href="https://arxiv.org/abs/2012.09164">Point Transformer</a> releasing their official implementation, training pipelines and trained weights.

