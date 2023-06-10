import math
import numpy as np
import random


def arrange_dataset_by_target(targets: np.ndarray, **kwargs):
    """
    :param targets: targets in the original dataset
    :param kwargs:
        long_tail: whether to make long-tailed dataset;
        gamma: gamma coefficient in making long-tailed dataset;
        shuffle: whether to shuffle samples of all classes
    :return: (dict) indices of all classes
    """
    long_tail = kwargs.get('long_tail', False)
    gamma = kwargs.get('gamma', 0.9)
    gamma = min(max(gamma, 0), 1)
    shuffle = kwargs.get('shuffle', False)

    # generate samples_by_class
    classes = np.unique(targets)
    samples_by_class = {}
    for c in classes:
        samples = np.where(np.equal(targets, c))[0]
        samples_by_class[c] = samples

    # make long-tailed dataset
    if long_tail:
        sample_num = [x.shape[0] for x in samples_by_class.values()]
        sample_num = np.array(sample_num)
        sort_idx = sample_num.argsort()
        sort_idx = np.flip(sort_idx)
        sorted_sample_num = sample_num[sort_idx]
        exp_sample_num = np.power(gamma, np.arange(len(classes)))
        base_rate = np.min(sorted_sample_num / exp_sample_num)
        final_sample_num = np.round(exp_sample_num * base_rate).astype(np.int64)
        fsn_cp = final_sample_num.copy()
        fsn_cp[sort_idx] = final_sample_num
        final_sample_num = fsn_cp
        ci2c = [x for x in samples_by_class.keys()]
        c2ci = {x: i for i, x in enumerate(ci2c)}
        for c, samples in samples_by_class.items():
            samples = np.random.choice(samples, final_sample_num[c2ci[c]], replace=False)
            samples_by_class[c] = samples

    # shuffle
    if shuffle:
        for c, samples in samples_by_class.items():
            np.random.shuffle(samples)
            samples_by_class[c] = samples

    return samples_by_class


def divide_iid(samples_by_class: dict, num_clients):
    samples_by_client = {}
    for cti in range(num_clients):
        samples = []
        for cs in samples_by_class.values():
            len_cs = cs.shape[0]
            s = int(cti / num_clients * len_cs)
            e = int((cti + 1) / num_clients * len_cs)
            samples.append(cs[s:e])
        samples = np.hstack(samples) if len(samples) > 0 else None
        samples_by_client[cti] = samples

    # check if any client has no sample
    for samples in samples_by_client.values():
        if len(samples) == 0:
            raise RuntimeError('Error when dividing dataset with iid method: number of samples is too few, '
                               'so it is unable to ensure that every client is assigned at least one sample. '
                               'Please make sure your dataset is big enough.')

    return samples_by_client


def divide_non_iid(samples_by_class: dict, num_clients, **kwargs):
    classes_per_client = kwargs.get('classes_per_client', 1)
    num_classes = len(samples_by_class)
    classes_per_client = min(classes_per_client, num_classes)

    seed = kwargs.get('seed', 10)
    prev_state = random.getstate()
    random.seed(seed)

    total_sample_num = sum(x.shape[0] for x in samples_by_class.values())
    if total_sample_num < num_clients:
        raise RuntimeError('Error when dividing dataset with non-iid method: number of samples is too few, '
                           'so it is unable to ensure that every client is assigned at least one sample. '
                           'Please make sure your dataset is big enough.')

    if num_clients * classes_per_client < num_classes:
        raise RuntimeError('Error when dividing dataset with non-iid method: number of clients is too few, '
                           f'so it is unable to ensure that every client is assigned exactly {classes_per_client} '
                           f'classes. Please set the number of clients to at least '
                           f'{math.ceil(num_classes / classes_per_client)}, or increase classes_per_client to '
                           f'at least {math.ceil(num_classes / num_clients)}.')

    client2classes = []
    sample_cnt = [0] * num_classes
    smaller_set = {i for i in range(num_classes)}
    for _ in range(num_clients):
        if len(smaller_set) >= classes_per_client:
            sampled = random.sample(smaller_set, classes_per_client)
            smaller_set -= set(sampled)
            if len(smaller_set) == 0:
                smaller_set = {i for i in range(num_classes)}
        else:
            bigger_set = {i for i in range(num_classes)} - smaller_set
            sampled_bigger = random.sample(
                bigger_set, classes_per_client - len(smaller_set))
            sampled = list(smaller_set) + sampled_bigger
            smaller_set = bigger_set - set(sampled_bigger)
        client2classes.append(sampled)
        for c in sampled:
            sample_cnt[c] += 1

    samples_by_client = {}
    class_sample_num = [x.shape[0] for x in samples_by_class.values()]
    class_sampled = [0] * num_classes
    for cti in range(num_clients):
        samples = []
        for c in client2classes[cti]:
            s_cnt = sample_cnt[c]
            c_sampled_cnt = class_sampled[c]
            c_sample_num = class_sample_num[c]
            c_sampled = round(c_sampled_cnt / s_cnt * c_sample_num)
            c_to_sample = round((c_sampled_cnt + 1) / s_cnt * c_sample_num)
            cs = samples_by_class[c]
            samples.append(cs[c_sampled:c_to_sample])
            class_sampled[c] += 1
        samples = np.hstack(samples) if len(samples) > 0 else None
        samples_by_client[cti] = samples

    random.setstate(prev_state)
    return samples_by_client


def non_iid_hard_sample(samples_by_class: dict, client_sample_num, tau):
    """
    Sample for non-iid-hard setting. This method can be, to some extent, regarded as "natural sampling".
    :param samples_by_class: samples by class
    :param client_sample_num: len(client_sample_num) = num_clients
    :param tau: sample at most k samples for one client per time, k = min(client_sample_num) * tau
    :return: samples_by_client
    """
    ci2c = [x for x in samples_by_class.keys()]

    sample_num = [x.shape[0] for x in samples_by_class.values()]
    # begin from the clients with smaller sample nums
    cti_order = np.array(client_sample_num).argsort().tolist()
    shard = max(round(client_sample_num[cti_order[0]] * tau), 1)

    samples_by_client = {}
    rem_sample_num = np.array(sample_num)
    for cti in range(len(client_sample_num)):
        samples_by_client[cti] = []

    for cti in cti_order:
        # sample for client cti
        demand = client_sample_num[cti]
        while demand > 0:
            p = rem_sample_num / rem_sample_num.sum()
            ci = np.random.choice(np.arange(len(sample_num)), p=p)
            supply = min(shard, rem_sample_num[ci])
            supply = min(supply, demand)
            s = sample_num[ci] - rem_sample_num[ci]
            e = s + supply
            c = ci2c[ci]
            samples_by_client[cti].extend(samples_by_class[c][s:e])

            rem_sample_num[ci] -= supply
            demand -= supply

            if sum(rem_sample_num) <= 0:  # dataset exhausted
                break
        if sum(rem_sample_num) <= 0:
            no_sample_cnt = sum([len(samples) == 0 for samples in samples_by_client.values()])
            if no_sample_cnt > 0:
                print(f'WARNING: Some clients ({no_sample_cnt}/{len(client_sample_num)}) are not allocated with '
                      f'any samples. This should not happen.')
            break

    for cti, samples in samples_by_client.items():
        samples_by_client[cti] = np.array(samples) if len(samples) > 0 else np.empty(0, dtype=np.int64)

    return samples_by_client


def divide_non_iid_hard(samples_by_class: dict, num_clients, **kwargs):
    gamma = kwargs.get('gamma', 0.9)
    gamma = min(max(gamma, 0), 1)
    tau = kwargs.get('tau', 0.5)
    tau = max(tau, 0)

    total_sample_num = sum(x.shape[0] for x in samples_by_class.values())
    if total_sample_num < num_clients:
        raise RuntimeError('Error when dividing dataset with non-iid-hard method: number of samples is too few, '
                           'so it is unable to ensure that every client is assigned at least one sample. '
                           'Please make sure your dataset is big enough.')

    total_sample_num -= num_clients
    exp_sample_num = np.power(gamma, np.arange(num_clients - 1, -1, -1))
    cum_exp_sample_num = np.cumsum(exp_sample_num)
    base_rate = total_sample_num / cum_exp_sample_num[-1]
    final_sample_num = np.round(cum_exp_sample_num * base_rate).astype(np.int64)
    final_sample_num[1:] = final_sample_num[1:] - final_sample_num[:-1]
    final_sample_num += 1

    return non_iid_hard_sample(samples_by_class, final_sample_num, tau)


def divide(method: str, samples_by_class, num_clients, **kwargs):
    if method == 'iid':
        samples_by_client = divide_iid(samples_by_class, num_clients)
    elif method == 'non-iid':
        samples_by_client = divide_non_iid(samples_by_class, num_clients, **kwargs)
    elif method == 'non-iid-hard':
        samples_by_client = divide_non_iid_hard(samples_by_class, num_clients, **kwargs)
    else:
        raise NotImplementedError(f'Method "{method}" not implemented.')

    return samples_by_client


def pollute_label(divisions: dict,
                  num_classes: int,
                  num_polluted_clients: int,
                  num_polluted_classes: int,
                  p_range=(0.5, 0.5)):
    """
    An experimental interface, used to deliberately replace the original labels
    in the clients' train sets with incorrect ones, so as to create some
    malicious clients (attackers) in federated learning.
    """
    from datasets.mnist import MNIST, FashionMNIST
    from datasets.cifar import CIFAR10

    print(f'WARNING: polluting datasets with params: '
          f'num_classes={num_classes}, num_polluted_clients={num_polluted_clients},'
          f'num_polluted_classes={num_polluted_classes}, p_range={p_range}')
    cti_list = list(divisions.keys())
    num_clients = len(cti_list)
    num_polluted_clients = min(max(num_polluted_clients, 0), num_clients)
    polluted_clients = np.random.choice(cti_list, num_polluted_clients)

    assert 0 <= p_range[0] <= p_range[1] <= 1

    train_set = divisions[0].dataset
    assert train_set.__class__ in [MNIST, FashionMNIST, CIFAR10], f'not implemented for {train_set.__class__}'

    for cti in polluted_clients:
        division = divisions[cti]
        div_sbc = division.samples_by_class
        num_div_classes = len(div_sbc)
        num_polluted_classes = min(max(num_polluted_classes, 0), num_div_classes)
        polluted_classes = np.random.choice(list(div_sbc.keys()), num_polluted_classes)
        for cls in polluted_classes:
            samples_idx = div_sbc[cls]
            p1, p2 = p_range
            if p1 == p2:
                p = p1
            else:
                p = np.random.rand() * (p2 - p1) + p1
            polluted = np.random.choice(samples_idx, round(len(samples_idx) * p)).tolist()

            # perform directed pollution
            different_classes = [c for c in range(num_classes) if c != cls]
            pollutant = np.random.choice(different_classes).tolist()
            for pi in polluted:
                if isinstance(train_set, (MNIST, FashionMNIST, CIFAR10)):
                    train_set.targets[pi] = pollutant

    print('Pollution complete!')
