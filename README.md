# [Continual Learning with Dynamic Sparse Training:Exploring Algorithms for Effective Model Updates](https://arxiv.org/abs/2308.14831)

## Training

Here, we provide parsing examples for training CL-with-DST.

To train 5-Task CIFAR100 *(total number of classses: 100, number of classes per task: 20)*
with uniform initialization, 
momentum growth,
90% sparsity,
100 epoch per task 
while updating the sparse topology every 100 batch:
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

To train 10-Task CIFAR100 *(total number of classses: 100, number of classes per task: 10)*
with ERK initialization, 
random growth,
80% sparsity,
100 epoch per task 
while updating the sparse topology every 400 batch:
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

To train 20-task CIFAR100 *(total number of classses: 100, number of classes per task: 5)*
with uniform initialization, 
unfired growth,
95% sparsity,
25 epoch per task 
while updating the sparse topology every 100 batch:
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

To train 10-Task miniImageNet *(total number of classses: 100, number of classes per task: 10)*
with ERK initialization, 
gradient growth,
80% sparsity,
100 epoch per task 
while updating the sparse topology every 400 batch:
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
You can download it from [kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) and place the .zip file named as *miniImageNet.zip* under the **data** folder.
