# Single-file Pytorch RegNet, with pretrained weights  

A single-file, modularized implementation of RegNet 
as introduced in
[\[I. Radosavovic, R. P. Kosaraju, R. Girshick, K. He, and P. Dollár\]: Designing Network Design Spaces. (CVPR), 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf)

Original implementation: [facebookresearch/pycls](https://github.com/facebookresearch/pycls)

Pretrained weights are taken from original implementation.

#### Usage  
The example below creates an RegNetY-200MF
model that takes 3-channel RGB image 
as input and outputs distribution over 50 classes, 
model weights are initialized
with weights pretrained on ImageNet dataset:
```python
from regnet import RegNet

model = RegNet('RegNetY-200MF',
                in_channels=3,
                num_classes=50,
                pretrained=True
                )
# inp - tensor of shape [batch_size, in_channels, image_height, image_width]
inp = torch.randn([10, 3, 224, 224])

# to get predictions:
pred = model(inp)
print('out shape:', pred.shape)
# >>> out shape: torch.Size([10, 50])

# to extract features:
features = model.get_features(inp)
for i, feature in enumerate(features):
    print('feature %d shape:' % i, feature.shape)
# >>> feature 0 shape: torch.Size([10, 24, 56, 56])
# >>> feature 1 shape: torch.Size([10, 56, 28, 28])
# >>> feature 2 shape: torch.Size([10, 152, 14, 14])
# >>> feature 3 shape: torch.Size([10, 368, 7, 7])

```
```pred``` is now a tensor containing output logits while 
```features``` is a list of 4 tensors representing outputs
of model's 4 intermediate convolutional stages.
 
#### Parameters
 
* ***name***, *(str)* - Model name, e.g. 'RegNetY-200MF'
* ***in_channels***, *(int)*, *(Default=3)* - Number of channels in input image
* ***num_classes***, *(int)*, *(Default=1000)* - Number of 
output classes
* ***bn_eps***, *(float)*, 
*(Default=1e-5)* - Batch normalizaton epsilon
* ***bn_momentum***, *(float)*, 
*(Default=0.1)* - Batch normalization momentum
* ***relu_inplace***, *(bool)*, 
*(Default=False)* - relu_inplace parameter for ReLU activation 
* ***pretrained***, *(bool)*, 
*(Default=False)* - Whether to initialize model with weights 
pretrained on ImageNet dataset
* ***progress***, *(bool)*, 
*(Default=False)* - Show progress bar when downloading 
pretrained weights.

#### Evaluation
A simple script to evaluate pretrained models against Imagenet 
validation set is provided in [imagenet_eval.ipynb](imagenet_eval.ipynb).

Accuracy achieved by models with pre-trained weights against ImageNet
dataset:

| Model | Accuracy, % |
| --- | --- |
| RegNetX-200MF | 69.718 |
| RegNetX-400MF | 73.47 |
| RegNetX-600MF | 74.754 |
| RegNetX-800MF | 75.662 |
| RegNetX-1.6GF | 77.632 |
| RegNetX-3.2GF | 78.85 |
| RegNetX-4.0GF | 79.06 |
| RegNetX-6.4GF | 79.556 |
| RegNetX-8.0GF | 79.784 |
| RegNetX-12GF | 80.138 |
| RegNetX-16GF | 80.56 |
| RegNetX-32GF | 80.776 |
|  |  |
| RegNetY-200MF | 71.356 |
| RegNetY-400MF | 74.92 |
| RegNetY-600MF | 76.182 |
| RegNetY-800MF | 77.106 |
| RegNetY-1.6GF | 78.68 |
| RegNetY-3.2GF | 79.58 |
| RegNetY-4.0GF | 80.112 |
| RegNetY-6.4GF | 80.512 |
| RegNetY-8.0GF | 80.614 |
| RegNetY-12GF | 81.052 |
| RegNetY-16GF | 81.104 |
| RegNetY-32GF | 81.464 |

#### Requirements
* Python v3.5+
* Pytorch v1.4+