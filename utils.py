import logging
from PIL import Image
from models import ResNet18, LeNet5, MobileNetV2, VGG16, MobileNetFunc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import copy
import hashlib
import os
import shutil
import requests
import zipfile
import io
import random
import torch.backends.cudnn as cudnn


''' LOGGING '''
logger = None
def set_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}task_{2}_{3}density_{4}death_{5}growth_{6}.log'.format(args.dataset,
                                                                                     args.num_tasks,
                                                                                     args.net_name,
                                                                                     args.density,
                                                                                     args.death,
                                                                                     args.growth,
                                                                                     hashlib.md5(str(args_copy).encode(
                                                                                         'utf-8')).hexdigest()[:8])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

''' DATASET '''
class iMNIST:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        else:
            self.mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.mnist):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.mnist[self.split_datasets[self.task_id][idx]]


class iCIFAR10:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        else:
            self.cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.cifar10):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.cifar10[self.split_datasets[self.task_id][idx]]


class iCIFAR100:
    def __init__(self, train=True, transform=None, tasks=None):
        if train:
            self.cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        else:
            self.cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        self.task_id = 0
        self.task_labels = tasks
        self.split_datasets = [list() for _ in range(len(tasks))]

        for i, (_, label) in enumerate(self.cifar100):
            for task_id, task_labels in enumerate(self.task_labels):
                if label in task_labels:
                    self.split_datasets[task_id].append(i)

    def set_task(self, task_id):
        self.task_id = task_id

    def __len__(self):
        return len(self.split_datasets[self.task_id])

    def __getitem__(self, idx):
        return self.cifar100[self.split_datasets[self.task_id][idx]]


class iTinyImageNet(Dataset):
    def __init__(self, root_dir, train=True, transform=None, tasks=None):

        if not os.path.exists('tiny-imagenet-200'):
            print('Downloading the dataset...')
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()

        self.root_dir = root_dir
        self.DIR = os.path.join(root_dir, 'train') if train else os.path.join(root_dir, 'val')

        if not train:
            if os.path.isfile(os.path.join(self.DIR, 'val_annotations.txt')):
                fp = open(os.path.join(self.DIR, 'val_annotations.txt'), 'r')
                data = fp.readlines()
                val_img_dict = {}  # dict {.jpg:[class_name]}
                for line in data:
                    words = line.split('\t')
                    val_img_dict[words[0]] = words[1]
                fp.close()
                for img, folder in val_img_dict.items():
                    newpath = (os.path.join(self.DIR, folder, 'images'))
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    if os.path.exists(os.path.join(self.DIR, 'images', img)):
                        os.rename(os.path.join(self.DIR, 'images', img), os.path.join(newpath, img))
            if os.path.exists(os.path.join(self.DIR, 'images')):
                os.rmdir(os.path.join(self.DIR, 'images'))
            if os.path.exists(os.path.join(self.DIR, 'val_annotations.txt')):
                os.remove(os.path.join(self.DIR, 'val_annotations.txt'))

        self.transform = transform
        self.tasks = tasks
        self.task_id = 0
        self.classes = os.listdir(self.DIR)  # list [class_name]
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}  # dict {class_name:[class_id]}
        self.class_files = {class_id: os.listdir(os.path.join(self.DIR, class_name, 'images'))
                            for class_name, class_id in self.class_to_id.items()}  # dict {class_id:[.jpg]}
        self.task_imgs = {}  # dict {task_id:[.jpg]}
        self.task_class_ids = {}  # dict {task_id:[class_id]}
        for task_no, class_ids in enumerate(tasks):
            samples_list = []
            class_list = []
            for class_id in class_ids:
                samples_list.extend(self.class_files[class_id])
                class_list.extend([class_id] * len(self.class_files[class_id]))
            self.task_imgs[task_no] = samples_list
            self.task_class_ids[task_no] = class_list

    def __len__(self):
        return len(self.task_class_ids[self.task_id])

    def __getitem__(self, idx):
        img_name = self.task_imgs[self.task_id][idx]
        folder_name = next((self.classes[class_id] for class_id, imgs in self.class_files.items() if img_name in imgs), None)
        img_path = os.path.join(self.DIR, folder_name, 'images', img_name)
        image = Image.open(img_path)
        class_id = self.task_class_ids[self.task_id][idx]
        if self.transform:
            image= image.convert('RGB')
            image = self.transform(image)
        return image, class_id

    def set_task(self, task_id):
        self.task_id = task_id

class iMiniImageNet(Dataset):
    def __init__(self, root_dir, train=True, transform=None, tasks=None):
        self.root_dir = root_dir
        self.DIR = os.path.join(root_dir, 'train') if train else os.path.join(root_dir, 'validation')

        if not os.path.exists(self.root_dir):
            with zipfile.ZipFile('data/miniImageNet100.zip', "r") as zip_ref:
                # Replace "path/to/dataset" with the path where you want to extract the files
                zip_ref.extractall(self.root_dir)

        if not os.path.exists(os.path.join(self.root_dir, 'validation')):
            validation_dir = os.path.join(self.root_dir, 'validation')
            # validation_fraction = 0.20
            # Loop through each folder in the data directory
            for folder_name in os.listdir(self.root_dir):
                folder_path = os.path.join(self.root_dir, folder_name)
                if os.path.isdir(folder_path):
                    # Create a validation folder for this folder
                    validation_folder = os.path.join(validation_dir, folder_name)
                    os.makedirs(validation_folder, exist_ok=True)

                    # Get a list of all the files in the folder
                    file_names = os.listdir(folder_path)
                    num_validation_files = 100
                    # num_validation_files = int(len(file_names) * validation_fraction)
                    random.shuffle(file_names)
                    # Move the first num_validation_files to the validation folder
                    for file_name in file_names[:num_validation_files]:
                        src_path = os.path.join(folder_path, file_name)
                        dst_path = os.path.join(validation_folder, file_name)
                        shutil.move(src_path, dst_path)

        if not os.path.exists(os.path.join(self.root_dir, 'train')):
            train_dir = os.path.join(self.root_dir, 'train')
            # Create the train directory if it doesn't exist
            os.makedirs(train_dir, exist_ok=True)

            # Loop through each directory in the data directory
            for dir_name in os.listdir(self.root_dir):
                dir_path = os.path.join(self.root_dir, dir_name)
                if os.path.isdir(dir_path) and dir_name != "validation" and dir_name != "train":
                    # Move the directory to the train directory
                    new_dir_path = os.path.join(train_dir, dir_name)
                    shutil.move(dir_path, new_dir_path)

        self.transform = transform
        self.tasks = tasks
        self.task_id = 0
        self.classes = os.listdir(self.DIR)  # list [class_name]
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}  # dict {class_name:[class_id]}
        self.class_files = {class_id: os.listdir(os.path.join(self.DIR, class_name))
                            for class_name, class_id in self.class_to_id.items()}  # dict {class_id:[.jpg]}
        self.task_imgs = {}  # dict {task_id:[.jpg]}
        self.task_class_ids = {}  # dict {task_id:[class_id]}
        for task_no, class_ids in enumerate(tasks):
            samples_list = []
            class_list = []
            for class_id in class_ids:
                samples_list.extend(self.class_files[class_id])
                class_list.extend([class_id] * len(self.class_files[class_id]))
            self.task_imgs[task_no] = samples_list
            self.task_class_ids[task_no] = class_list

    def __len__(self):
        return len(self.task_class_ids[self.task_id])

    def __getitem__(self, idx):
        img_name = self.task_imgs[self.task_id][idx]
        folder_name = next((self.classes[class_id] for class_id, imgs in self.class_files.items() if img_name in imgs), None)
        img_path = os.path.join(self.DIR, folder_name, img_name)
        image = Image.open(img_path)
        class_id = self.task_class_ids[self.task_id][idx]
        if self.transform:
            image = image.convert('RGB')
            image = self.transform(image)
        return image, class_id

    def set_task(self, task_id):
        self.task_id = task_id


''' UTILS '''
def load_dataset(dataset, train=True, tasks=None):
    if dataset == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        dataset = iMNIST(train=train, transform=transform, tasks=tasks)
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = iCIFAR10(train=train, transform=transform, tasks=tasks)
    if dataset == 'cifar100':
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = iCIFAR100(train=train, transform=transform, tasks=tasks)
    if dataset == 'tiny-imagenet-200':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=train)
        dataset = iTinyImageNet('tiny-imagenet-200', train=train, transform=transform, tasks=tasks)
    if dataset == 'miniImagenet100':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        dataset = iMiniImageNet('data/miniImageNet100', train=train, transform=transform, tasks=tasks)

    return dataset


def get_transform(size, padding, mean, std, preprocess):
    transform = []
    transform.append(transforms.Resize(size))
    transform.append(transforms.RandomCrop(size, padding=4))
    transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform)


def grayscale_to_rgb(image):
    if image.size(0) == 1:
        return torch.cat((image, image, image), dim=0)
    else:
        return image


def get_loader(dataset, task_id, batch):
    dataset.set_task(task_id)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2)
    return loader


def get_network(net_name, num_classes, device):
    if net_name == 'resnet18':
        net = ResNet18(num_classes).to(device)
        NET = ResNet18(num_classes).to(device)
    elif net_name == 'lenet':
        net = LeNet5(num_classes).to(device)
        NET = LeNet5(num_classes).to(device)
    elif net_name == 'mobilenetv2':
        net = MobileNetV2(num_classes).to(device)
        NET = MobileNetV2(num_classes).to(device)   
    elif net_name == 'vgg16':
        net = VGG16('like', num_classes).to(device)
        NET = VGG16('like', num_classes).to(device)
    return net, NET


def set_optimizer(optimizer, lr, momentum, l2, epochs, net):
    if optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=l2, nesterov=True)
    elif optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)
    else:
        print('Unknown optimizer: {0}'.format(optimizer))
        raise Exception('Unknown optimizer.')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[int(epochs * 1 / 2),
                                                                    int(epochs * 3 / 4)],
                                                        gamma=0.1)
    return optimizer, lr_scheduler


def save_checkpoint(net, dataset, net_name):
    print_and_log('Saving Model.\n')
    torch.save(net.state_dict(), './models/{0}_{1}.pt'.format(dataset, net_name))


def save_mask(mask, dataset, task_id, net_name):
    print_and_log('Saving Mask.\n')
    torch.save(mask.masks, './masks/{0}_task{1}_{2}_mask.pt'.format(dataset, task_id, net_name))


def set_args(args):

    if not os.path.exists('./models'): os.mkdir('./models')
    if not os.path.exists('./logs'): os.mkdir('./logs')
    if not os.path.exists('./masks'): os.mkdir('./masks')

    set_logger(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print_and_log(args)
    print_and_log('=' * 60)
    print_and_log(
        'Dataset: {0}, #Task:{1}, #ClassPerTask:{2}'.format(args.dataset, args.num_tasks,
                                                            args.num_classes_per_task))
    print_and_log('Model: {0}'.format(args.net_name))
    print_and_log('=' * 60)
    print_and_log('=' * 60)
    if args.sparse:
        print_and_log('Init mode: {0}'.format(args.init))
        print_and_log('Death mode: {0}'.format(args.death))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)


''' CONTINUAL LEARNING '''
def create_labels(num_classes, num_tasks, num_classes_per_task):
    tasks_order = np.arange(num_classes)
    labels = tasks_order.reshape((num_tasks, num_classes_per_task))

    return labels


def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    for labels in task_labels:
        idx = np.in1d(dataset.targets, labels)
        splited_dataset = copy.deepcopy(dataset)
        splited_dataset.targets = torch.tensor(splited_dataset.targets)[idx]
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
    return datasets


def freeze_used_params(used_params, dataset, task_id, net_name):
    previous_mask = torch.load(f'./masks/{dataset}_task{task_id - 1}_{net_name}_mask.pt')
    used_params = {k: used_params.get(k, 0) + previous_mask.get(k, 0) for k in
                   set(used_params) | set(previous_mask)}
    return used_params


def set_mask(dataset, net_name, NET, task_id):
    task_mask = torch.load(f'./masks/{dataset}_task{task_id}_{net_name}_mask.pt')
    masked_net = copy.deepcopy(NET)

    for n, t in masked_net.named_parameters():
        if n in task_mask:
            t.data = t.data * task_mask[n]
    return masked_net


def ewc(args, net, train_loader, task_id, fisher_dict, optpar_dict, device):
    net.train()
    optimizer_ewc = optim.Adam(net.parameters(), lr=args.lr)
    optimizer_ewc.zero_grad()

    # accumulating gradients
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in net.named_parameters():
        optpar_dict[task_id][name] = param.data.clone()
        fisher_dict[task_id][name] = param.grad.data.clone().pow(2)


def subnet_selection(net, test_set, task_id, device):
    max_out = []
    for t in range(task_id + 1):
        test_loader = torch.utils.data.DataLoader(test_set[t],
                                                  batch_size=10,
                                                  shuffle=True,
                                                  num_workers=2)
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)

        output = net(data)
        max_out.append(torch.max(output, dim=1)[0].sum().cpu().detach())
    j0 = np.argmax(max_out)
    return j0
