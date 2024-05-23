# Continual Learning with Dynamic Sparse Training: Exploring Algorithms for Effective Model Updates 

*CL-with-DST is the first empirical study investigating the effect of different Dynamic Sparse Training (DST) components in Continual learning (CL).*

## Training

Here, we provide parsing examples for training CL-with-DST.

To train 10-Task CIFAR100 *(total number of classes: 100, number of classes per task: 10)*
with ERK initialization, 
random growth,
80% sparsity,
100 epoch per task 
while updating the sparse topology every 400 batches:
```
python cifar100.py
       --dataset cifar100
       --num_tasks 10
       --num_classes 100
       --num_classes_per_task 10
       --sparse_init ERK 
       --growth random  
       --density 0.2 
       --epochs 100
       --update_frequency 400 
```

To train 10-Task miniImageNet *(total number of classes: 100, number of classes per task: 10)*
with ERK initialization, 
gradient growth,
80% sparsity,
100 epoch per task 
while updating the sparse topology every 400 batches:
```
python miniImageNet100.py
       --dataset miniImageNet
       --num_tasks 10
       --num_classes 100
       --num_classes_per_task 10
       --sparse_init ERK 
       --growth gradient  
       --density 0.2 
       --epochs 100
       --update_frequency 400 
```
More options and explanations can be found in the ArgumentParser().

Note: You should download the miniImageNet dataset to run experiments on that. 
You can download it from [kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) and place the .zip file named *miniImageNet.zip* under the **data** folder.
