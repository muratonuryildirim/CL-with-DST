import numpy as np
import torch
import copy
import warnings
import argparse
from core import Masking, train, evaluate, DeathCosineDecay
from utils import set_args, print_and_log, set_optimizer, set_mask, create_labels, load_dataset,\
                  get_network, get_loader, freeze_used_params, save_mask, save_checkpoint


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='miniImagenet100',
                        help='dataset to use')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='number of classes')
    parser.add_argument('--num_tasks', type=int, default=10,
                        help='number of tasks')
    parser.add_argument('--num_classes_per_task', type=int, default=10,
                        help='number of classes per task')
    parser.add_argument('--net_name', type=str, default='resnet18',
                        help='network architecture to use')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer. options: adam, sgd')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='sgd momentum')
    parser.add_argument('--l2', type=float, default=0,
                        help='weight decay coefficient')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of examples per training batch')
    parser.add_argument('--sparse', type=bool, default=True,
                        help='enable sparse mode. Default: True.')
    parser.add_argument('--density', type=float, default=0.1,
                        help='density of the overall sparse network.')
    parser.add_argument('--init', type=str, default='uniform',
                        help='sparse initialization. options: ERK, uniform')
    parser.add_argument('--death', type=str, default='magnitude',
                        help='pruning mode. options: magnitude, SET, Taylor_FO.')
    parser.add_argument('--growth', type=str, default='random',
                        help='rewiring mode. options: random, random_unfired, gradient and momentum.')
    parser.add_argument('--death_rate', type=float, default=0.50,
                        help='pruning rate.')
    parser.add_argument('--death_growth_ratio', type=float, default=1.0,
                        help='ratio between pruning and rewiring.')
    parser.add_argument('--redistribution', type=str, default='none',
                        help='redistribution mode. options: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--update_frequency', type=int, default=50,
                        help='number of batches to train between parameter exploration')
    parser.add_argument('--mask_selection', type=str, default='max',
                        help='mask selection method. options: max')
    parser.add_argument('--isolate', type=bool, default=True,
                        help='freeze weights of the previous tasks')
    parser.add_argument('--regularize', type=bool, default=False,
                        help='regularize weights of the previous tasks')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--train', type=bool, default=True,
                        help='train the model; if False inference mode only')
    parser.add_argument('--test', type=bool, default=True,
                        help='test the model; if False train mode only')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    warnings.filterwarnings("ignore", category=UserWarning)

    set_args(args)

    task_list = create_labels(args.num_classes, args.num_tasks, args.num_classes_per_task)
    train_dataset = load_dataset(args.dataset, train=True, tasks=task_list)
    test_dataset = load_dataset(args.dataset, train=False, tasks=task_list)
    net, NET = get_network(args.net_name, args.num_classes, device)

    if args.train:
        used_params = {}
        mask = None
        for task_id in range(args.num_tasks):
            train_loader = get_loader(train_dataset, task_id=task_id, batch=args.batch_size)
            optimizer, lr_scheduler = set_optimizer(args.optimizer, args.lr, args.momentum, args.l2,
                                                    args.epochs, net)

            if args.sparse:
                if task_id != 0:
                    net.load_state_dict(
                        torch.load('./models/{0}_{1}_DENSE.pt'.format(args.dataset, args.net_name)))
                    if args.isolate:
                        used_params = freeze_used_params(used_params, args.dataset, task_id,
                                                         args.net_name)
                death_rate_decay = DeathCosineDecay(args.death_rate,
                                                    len(train_loader) * args.epochs)
                mask = Masking(args.init, args.density, args.death, args.growth,
                               args.death_rate, args.epochs, args.update_frequency,
                               args.death_growth_ratio,
                               death_rate_decay, optimizer, device)
                mask.add_module(net)

            print_and_log('TRAINING...\n')
            best_acc = -1.0
            for epoch in range(args.epochs):
                train_loss, train_acc = train(net, optimizer, task_id, train_loader,
                                              args.num_classes_per_task, args.regularize,
                                              args.isolate,
                                              args.batch_size, epoch, device, mask, used_params)
                lr_scheduler.step()
                if train_acc > best_acc:
                    save_checkpoint(net, args.dataset, args.net_name)
                    best_acc = train_acc
                    if args.sparse:
                        save_mask(mask, args.dataset, task_id, args.net_name)

            if args.sparse:
                net.load_state_dict(
                    torch.load('./models/{0}_{1}.pt'.format(args.dataset, args.net_name)))
                if task_id == 0:
                    NET = copy.deepcopy(net)
                else:
                    for (name, param), (old_name, old_param) in zip(net.named_parameters(),
                                                                    NET.named_parameters()):
                        param.data[param == 0] = old_param.data[param == 0]
                        NET = copy.deepcopy(net)
                torch.save(NET.state_dict(),
                           './models/{0}_{1}_DENSE.pt'.format(args.dataset, args.net_name))

            test_loader = get_loader(test_dataset, task_id=task_id, batch=args.batch_size)
            masked_net = set_mask(args.dataset, args.net_name, NET, task_id)
            _, _, test_loss, test_accuracy = evaluate(masked_net, device, test_loader,
                                                      args.num_classes_per_task, task_id)
          
    if args.test:
        print_and_log('TESTING...\n')
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        bwt = np.zeros(args.num_tasks)
        fwt = np.zeros(args.num_tasks)
        if args.sparse:
            NET.load_state_dict(
                torch.load('./models/{0}_{1}_DENSE.pt'.format(args.dataset, args.net_name)))
        else:
            net.load_state_dict(
                torch.load('./models/{0}_{1}.pt'.format(args.dataset, args.net_name)))

        for task_id in range(args.num_tasks):
            for t in range(args.num_tasks):
                test_loader = get_loader(test_dataset, task_id=t, batch=args.batch_size)
                if args.sparse:
                    if task_id >= t:
                        masked_net = set_mask(args.dataset, args.net_name, NET, t)
                        _, _, test_loss, test_accuracy = evaluate(masked_net, device, test_loader,
                                                                  args.num_classes_per_task,
                                                                  task_id)
                    else:
                        masked_net = set_mask(args.dataset, args.net_name, NET, task_id)
                        _, _, test_loss, test_accuracy = evaluate(masked_net, device, test_loader,
                                                                  args.num_classes_per_task,
                                                                  task_id, t,
                                                                  is_fwt=True)
                acc_matrix[task_id, t] = test_accuracy * 100

        for task_id in range(1, args.num_tasks):
            for t in range(task_id):
                bwt[task_id] += (acc_matrix[task_id, t] - acc_matrix[t, t]) / task_id

        for task_id in range(args.num_tasks):
            for t in range(task_id + 1, args.num_tasks):
                fwt[task_id] = (np.mean(acc_matrix[task_id, t:args.num_tasks]))
                break

        final_acc = acc_matrix[-1]
        inc_acc = np.zeros_like(final_acc)
        for i in range(len(final_acc)):
            if i == 0:
                inc_acc[i] = final_acc[i]
            else:
                current_values = final_acc[:i + 1]
                inc_acc[i] = np.mean(current_values)

        print_and_log('ACCURACY MATRIX:\n {}\n'.format(acc_matrix))
        print_and_log('INCREMENTAL ACCURACY:\n {}\n'.format(inc_acc))
        print_and_log('BWT:\n {}\n'.format(bwt))
        print_and_log('FWT:\n {}\n'.format(fwt))
        print_and_log('Average BWT:\n {}\n'.format(bwt[-1]))
        print_and_log('Average FWT:\n {}\n'.format(np.mean(fwt[:-1])))


if __name__ == '__main__':
    main()
