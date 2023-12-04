# [CPAL-2024] Continual Learning with Dynamic Sparse Training: Exploring Algorithms for Effective Model Updates 

*Continual learning (CL) refers to the ability of an intelligent system to sequentially acquire
and retain knowledge from a stream of data with as little computational overhead as possible.
To this end; regularization, replay, architecture, and parameter isolation approaches were
introduced to the literature. Parameter isolation using a sparse network which enables
to allocate distinct parts of the neural network to different tasks and also allows to share
of parameters between tasks if they are similar. Dynamic Sparse Training (DST) is a
prominent way to find these sparse networks and isolate them for each task. This paper is
the first empirical study investigating the effect of different DST components under the CL
paradigm to fill a critical research gap and shed light on the optimal configuration of DST
for CL if it exists. Therefore, we perform a comprehensive study in which we investigate
various DST components to find the best topology per task on well-known CIFAR100 and
miniImageNet benchmarks in a task-incremental CL setup since our primary focus is to
evaluate the performance of various DST criteria, rather than the process of mask selection.
We found that, at a low sparsity level, Erd ̋os-Rényi Kernel (ERK) initialization utilizes the
backbone more efficiently and allows to effectively learn increments of tasks. At a high
sparsity level, unless it is extreme, uniform initialization demonstrates more reliable and
robust performance. In terms of growth strategy; performance is dependent on the defined
initialization strategy and the extent of sparsity. Finally, adaptivity within DST components
is a promising way for better continual learners.* ([click](https://arxiv.org/abs/2308.14831) to reach the full paper)

## Training

Here, we provide parsing examples for training CL-with-DST.

To train 5-Task CIFAR100 *(total number of classes: 100, number of classes per task: 20)*
with uniform initialization, 
momentum growth,
90% sparsity,
100 epoch per task 
while updating the sparse topology every 100 batches:
```
python cifar100.py
       --dataset cifar100
       --num_tasks 5
       --num_classes 100
       --num_classes_per_task 20
       --sparse_init uniform 
       --growth momentum  
       --density 0.1 
       --epochs 100
       --update_frequency 100 
```

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

To train 20-task CIFAR100 *(total number of classes: 100, number of classes per task: 5)*
with uniform initialization, 
unfired growth,
95% sparsity,
25 epochs per task 
while updating the sparse topology every 100 batches:
```
python cifar100.py
       --dataset cifar100
       --num_tasks 20
       --num_classes 100
       --num_classes_per_task 5
       --sparse_init uniform 
       --growth unfired  
       --density 0.05 
       --epochs 25
       --update_frequency 100 
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
