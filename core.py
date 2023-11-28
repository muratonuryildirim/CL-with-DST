import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
from utils import print_and_log


class DeathCosineDecay:
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']


class DeathLinearDecay:
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate * self.factor
        else:
            return death_rate


class Masking:
    def __init__(self, sparse_init, density, death_mode, growth_mode, death_rate, epochs,
                 update_frequency, death_growth_ratio, death_rate_decay, optimizer, device):
        growth_modes = ['random', 'random_unfired', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))


        self.device = device
        self.sparse_init = sparse_init
        self.density = density
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.death_growth_ratio = death_growth_ratio
        self.death_rate_decay = death_rate_decay

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        #GMP
        self.init_death_epoch = epochs/2
        self.final_death_epoch = epochs - (epochs/10)

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.death_rate = death_rate
        self.baseline_nonzero = None
        self.steps = 0
        self.death_every_k_steps = update_frequency

    def init(self, sparse_init, density, erk_power_scale=1.0):
        self.density = density
        if sparse_init == 'GMP':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).to(self.device)
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        elif sparse_init == 'lottery_ticket':
            print('initialize by lottery ticket')
            self.baseline_nonzero = 0
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * self.density)

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        elif sparse_init == 'uniform':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.to(self.device)  # lsw
                    self.baseline_nonzero += weight.numel() * density

        elif sparse_init == 'ERK':
            #print('initialize by ERK')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - self.density)
                    n_ones = n_param * self.density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            #print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                #print(f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}")
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.to(self.device)

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks)  # used for ITOP
        # self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        #print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        self.death_rate = self.death_rate_decay.get_dr()
        self.steps += 1

        if self.death_every_k_steps is not None:
            if self.steps % self.death_every_k_steps == 0:
                self.truncate_weights()
                _, _ = self.fired_masks_update()
                self.print_nonzero_counts()

    def add_module(self, module):
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)

        #print('Removing biases...')
        self.remove_weight_partial_name('bias')
        #self.remove_weight_partial_name('se')
        #print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        #print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.init(sparse_init=self.sparse_init, density=self.density)

        return self.modules

    def remove_weight(self, name):
        if name in self.masks:
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            self.masks.pop(name + '.weight')

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                removed.add(name)
                self.masks.pop(name)

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]
                    # reset momentum
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor][
                                                                              'momentum_buffer'] * self.masks[name]

    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        death_rate = 1 - self.density
        curr_death_epoch = epoch
        total_death_epochs = self.final_death_epoch - self.init_death_epoch + 1
        if epoch >= self.init_death_epoch and epoch <= self.final_death_epoch:
            death_decay = (1 - ((curr_death_epoch - self.args.init_death_epoch) / total_death_epochs)) ** 3
            curr_death_rate = death_rate - (death_rate * death_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_death_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density,
                                                                                            sparse_size / total_size,
                                                                                            epoch))

    def truncate_weights(self):

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                # death
                if self.death_mode == 'magnitude':
                    new_mask = self.magnitude_death(mask, weight, name)
                elif self.death_mode == 'SET':
                    new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = self.taylor_FO(mask, weight, name)
                elif self.death_mode == 'threshold':
                    new_mask = self.threshold_death(mask, weight, name)

                self.num_remove[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                self.masks[name][:] = new_mask

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name, new_mask, weight)

                if self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name, new_mask, weight)

                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

        self.apply_mask()

    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k - 1].item()

        return (torch.abs(weight.data) > threshold)

    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k - 1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove / 2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k - 1].item()

        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name] == 0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name] == 0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth - num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).to(self.device) < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth / n)
        new_weights = torch.rand(new_mask.shape).to(self.device) < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_ != 0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)
        grad = grad * (new_mask == 0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def momentum_neuron_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2:
            sum_dim = [1]
        elif len(M.shape) == 4:
            sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask == 0).sum(sum_dim)

        M = M * (new_mask == 0).float()
        for i, fraction in enumerate(v):
            neuron_regrowth = math.floor(fraction.item() * total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''

    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1 / (torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
                                                               num_nonzeros / float(mask.numel()))
                print(val)

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item()) / float(
                    self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights / ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights


def train(net, optimizer, task_id, train_loader, num_classes_per_task, regularize, isolate,
          batch_size, epoch, device, mask=None, used_params=None):
    t0 = time.time()
    net.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(torch.int64).to(device)
        net.to(device)
        output = net(data)
        offset_a = task_id * num_classes_per_task
        offset_b = (task_id + 1) * num_classes_per_task
        loss = F.cross_entropy(output[:, offset_a:offset_b], target - offset_a)

        if task_id != 0 and regularize:
            ewc_lambda = 100
            for task in range(task_id):
                for name, param in net.named_parameters():
                    fisher = fisher_dict[task][name]
                    optpar = optpar_dict[task][name]
                    loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if task_id != 0 and isolate:
            for name, params in net.named_parameters():
                if name in mask.masks:
                    params.grad[used_params[name] != 0] = 0

        if mask is not None: mask.step()
        else: optimizer.step()

        log_interval = 20
        if batch_idx % log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                          'Loss: {:.6f} Accuracy: {}/{} ({:.4f}%) '.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader) * batch_size,
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item(),
                                                                           correct,
                                                                           n,
                                                                           100. * correct / float(n)))
    train_acc = 100. * correct / float(n)
    print_and_log('\n{}:Accuracy: {}/{} ({:.4f}%)\n'
                  'Time taken for epoch: {:.2f} seconds.\n'.format('Training summary',
                                                                   correct,
                                                                   n,
                                                                   train_acc,
                                                                   time.time() - t0))
    return train_loss, train_acc


def evaluate(net, device, test_loader, num_classes_per_task, task_id, t=None, is_fwt=False):
    net.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            if is_fwt:
                target = target - ((t-task_id) * num_classes_per_task)
            data, target = data.to(device), target.to(torch.int64).to(device)
            # if fp16: data = data.half()
            output = net(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format('Test evaluation',
                                                                                 test_loss,
                                                                                 correct,
                                                                                 n,
                                                                                 100. * correct / float(n)))
    return correct, float(n), test_loss, correct / float(n)

