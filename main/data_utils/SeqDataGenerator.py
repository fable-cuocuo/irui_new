import random
import time
import multiprocessing as mp
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os

random.seed(0)


class SeqDataCollector(object):

    def __init__(self, config):
        print('#' * 10 + ' DataInfo ' + '#' * 10)
        self.input_len = config['input_len']

        self.candidate_num = config['candidate_num']

        self.device = config['device']
        self.data_path = './datasets/' + config['dataset'] + '/seq/'
        self.train_neg_num = config['train_neg_num']
        self.thread_num = config['thread_num']
        random.seed(123)
        np.random.seed(123)
        self.user2Idx = {}
        self.item2Idx = {'mask': 0}
        self.itemIdx2Str = {}
        # dict 内含user-item信息
        self.userIdx2sequence = {}
        self.userItemSet = {}
        self.itemUserList = {}
        self.item_freq = {}
        self.valid_users = set()
        self.valid_items = set()
        self.cpu_num = 0

        self.numUser = 0
        self.numItem = 0
        self.item_item_content = None
        self.item_item_interaction = None
        self.item_dist = [0 for _ in range(self.numItem)]
        self.user_dist = [0 for _ in range(self.numUser)]
        self.load_seq_data()

        config['user_num'] = self.numUser
        config['item_num'] = self.numItem
        print(f'numUser:{self.numUser}')
        print(f'numItem:{self.numItem}')
        self.eval_neg_num = config['eval_neg_num']
        self.train_batch_size = config['train_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.train_size = 0
        self.valid_size = 0
        self.test_size = 0

    def load_file(self, file_name):
        file_path = self.data_path + file_name
        line_count = 0
        if os.path.exists(file_path):
            print(f'reading {file_name}')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split(' ')
                    user, item = splited_line[0], splited_line[1]
                    if user not in self.user2Idx:
                        userIdx = len(self.user2Idx)
                        self.user2Idx[user] = userIdx
                    if item not in self.item2Idx:
                        itemIdx = len(self.item2Idx)
                        self.item2Idx[item] = itemIdx
                    userIdx = self.user2Idx[user]
                    itemIdx = self.item2Idx[item]
                    if userIdx not in self.userIdx2sequence:
                        self.userIdx2sequence[userIdx] = []
                    self.userIdx2sequence[userIdx].append(itemIdx)
                    line_count += 1
        return line_count

    def load_seq_data(self):
        full_seq_count = self.load_file('seq.dat')
        self.numUser = len(self.user2Idx)
        self.numItem = len(self.item2Idx)
        # build item distribution
        self.item_dist = [0 for _ in range(self.numItem)]
        self.user_dist = [0 for _ in range(self.numUser)]
        seq_len_sum = 0
        for user, item_seq in self.userIdx2sequence.items():
            seq_len_sum += len(item_seq)
            for item in item_seq:
                self.item_dist[item] += 1
            self.user_dist[user] = len(item_seq)
        assert seq_len_sum == full_seq_count
        self.generate_item_dist()
        for user, items in self.userIdx2sequence.items():
            self.userItemSet[user] = set(items)

    def generate_train_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating train samples')
        start = time.time()
        train_users = []
        train_hist_items = []
        train_masks = []  # mask是把非0的数字变为1
        train_targets = []
        # 新增candidate
        train_candidates = []

        abandon_count = 0
        candidate_num = self.candidate_num
        sub_seq_len = input_len + 1  # input+预测结果
        for user, item_full_seq in self.userIdx2sequence.items():  # 遍历用户字典
            if len(item_full_seq) < 7 :
                continue
            item_train_seq = item_full_seq[0:-1]  # 除了最后一个
            for sub_seq, mask, candidate_items in self.slide_window(item_train_seq, sub_seq_len, candidate_num):
                # input_len = 5 , candidate_num = 4, sub_seq_len = 10
                input_seq = sub_seq[0: input_len]
                input_mask = mask[0: input_len]
                target = sub_seq[input_len]  # “预测结果”
                assert len(sub_seq) == len(mask) == sub_seq_len
                # append lists
                train_users.append(user)
                train_hist_items.append(input_seq)  # list内的元素为list
                train_masks.append(input_mask)
                train_targets.append(target)
                train_candidates.append(candidate_items)
                self.valid_users.add(user)
                self.valid_items.add(target)
                for item in sub_seq:
                    if item is not 0:
                        self.valid_items.add(item)
        self.train_size = len(train_users)
        print(f"train_size: {self.train_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.train_size, 4)}) samples')
        print(f"valid user num: {len(self.valid_users)}")
        print(f"valid item num: {len(self.valid_items)}")

        dataset = UnidirectTrainDataset(train_users, train_hist_items,
                                        train_masks, train_targets, train_candidates, self.userItemSet,
                                        max_item_idx=self.numItem - 1, neg_num=self.train_neg_num,
                                        item_dist=self.item_dist)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=self.cpu_num, batch_size=self.train_batch_size)

        return dataloader

    def generate_valid_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating valid samples')
        start = time.time()
        valid_users = []
        valid_hist_items = []
        valid_masks = []
        valid_targets = []
        abandon_count = 0
        for user, item_full_seq in self.userIdx2sequence.items():
            if len(item_full_seq) < 7 :
                continue
            target_item = item_full_seq[-2]
            if user not in self.valid_users or target_item not in self.valid_items:
                abandon_count += 1
                continue
            valid_users.append(user)
            raw_input = item_full_seq[0:-2]
            raw_input_len = len(raw_input)
            if raw_input_len >= input_len:
                input = raw_input[-input_len:]
                mask = [1] * input_len
            else:  # raw_input_len < input_len
                input = [0] * (input_len - raw_input_len) + raw_input
                mask = [0] * (input_len - raw_input_len) + [1] * raw_input_len
            assert len(input) == len(mask) == input_len
            assert item_full_seq[-3] == input[-1]

            valid_hist_items.append(input)
            valid_masks.append(mask)
            valid_targets.append(target_item)
            candidate = []

        dataset = UnidirectTrainDataset(valid_users, valid_hist_items,
                                        valid_masks, valid_targets, candidate, self.userItemSet,
                                        max_item_idx=self.numItem - 1, neg_num=self.train_neg_num,
                                        item_dist=self.item_dist)
        dataloader = DataLoader(dataset, shuffle=True,
                                num_workers=self.cpu_num, batch_size=self.train_batch_size)
        self.valid_size = len(valid_users)
        print(f"valid_size: {self.valid_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.valid_size, 4)}) samples')
        return dataloader

    def generate_test_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating test samples')
        start = time.time()
        test_users = []
        test_hist_items = []
        test_masks = []
        test_targets = []
        abandon_count = 0
        for user, item_full_seq in self.userIdx2sequence.items():
            if len(item_full_seq) < 7:
                continue
            test_users.append(user)  # list，所有user
            raw_input = item_full_seq[0:-1]  # 第一个元素到倒数第二个元素
            raw_input_len = len(raw_input)
            if raw_input_len >= input_len:
                input = raw_input[-input_len:]  # 如果剪切下的len>input_len，那么就把后input_len个item剪切下来作为test的input
                mask = [1] * input_len
            else:  # raw_input_len < input_len
                input = [0] * (input_len - raw_input_len) + raw_input  # 如果小于的话，则在前面补0
                mask = [0] * (input_len - raw_input_len) + [1] * raw_input_len
            assert len(input) == len(mask) == input_len
            assert item_full_seq[-2] == input[-1]

            test_hist_items.append(input)
            test_masks.append(mask)

            sampled_negs = []
            while len(sampled_negs) < 100:
                sampled_neg_cands = np.random.choice(self.numItem, self.eval_neg_num, False, self.item_dist)  # 随机挑选负样本
                valid_neg_ids = [x for x in sampled_neg_cands if x not in self.userItemSet[user]]
                sampled_negs.extend(valid_neg_ids[:])
            sampled_negs = sampled_negs[:100]  # 99个负样本
            test_targets.append([item_full_seq[-1]] + list(sampled_negs))  # 长度为100

        dataset = UnidirectTestDataset(test_users, test_hist_items, test_masks, test_targets)
        dataloader = DataLoader(dataset, shuffle=False,
                                num_workers=self.cpu_num, batch_size=self.test_batch_size)  # 不打乱
        self.test_size = len(test_users)
        print(f"test_size: {self.test_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.test_size, 4)}) samples')
        return dataloader

    def generate_item_dist(self):
        # print('generating item distribution')
        item_dist = np.array(self.item_dist)
        sum_click = item_dist.sum()
        self.item_dist = item_dist / sum_click
        # print(f'item dist 0: {self.item_dist[0]}')
        self.item_dist[0] = 0


    def slide_window(self, itemList, window_size, candidate_num):
        """
        Input a sequence [1, 2, 3, 4, 5] with window size 3
        Return [1, 2, 3],  [2, 3, 4],  [3, 4, 5]
        with   [1, 1, 1],  [1, 1, 1],  [1, 1, 1]

        Or a sequence [1, 2, 3, 4, 5] with window size 7
        Return [0, 0, 1, 2, 3, 4, 5]
        with   [0, 0, 1, 1, 1, 1, 1]
        """
        # 保证input里最起码有两个item
        new_item_list = [0] * (window_size - 2) + itemList
        # print(new_item_list)
        num_seq = len(new_item_list) - window_size + 1
        # print(f"{len(new_item_list)} and {num_seq}")
        assert num_seq == len(itemList) - 1
        # 加入candidate []前闭后开
        for startIdx in range(num_seq):
            endIdx = startIdx + window_size
            item_sub_seq = new_item_list[startIdx: endIdx]
            # print(f"start: {startIdx} end: {endIdx}")
            can_end = endIdx + candidate_num
            if can_end < len(new_item_list):
                candidate = new_item_list[endIdx: can_end]
            else:
                candidate = new_item_list[endIdx:]
                zero_num = candidate_num - len(candidate)
                sample_candidate = np.random.choice(self.numItem, zero_num, False, self.item_dist)
                sample_candidate = sample_candidate.tolist()
                # random_items = random.sample(range(1, numItem + 1), zero_num)
                # candidate.extend(random_items)
                candidate.extend(sample_candidate)
                '''
                length = len(new_item_list)
                num = candidate_num - length
                sampled = generate_sample_candidate(new_item_list, num)
                candidate.extend(sampled)
                '''
            mask = [1] * window_size
            for i, item in enumerate(item_sub_seq):
                if item is 0:
                    mask[i] = 0
            yield item_sub_seq, mask, candidate


class UnidirectTrainDataset(torch.utils.data.Dataset):

    def __init__(self, train_users, train_hist_items,
                 train_masks, train_targets, train_candidates, userItemSet, max_item_idx, neg_num, item_dist,
                 aug_num=3):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        train_masks = [bs, seq_len]
        train_targets = [bs]
        clean_mask = [bs], denoting whether this instance is clean or not
        """
        assert len(train_users) == len(train_hist_items) == len(train_masks) == len(train_targets)
        self.train_users = train_users
        self.train_hist_items = train_hist_items
        self.train_masks = train_masks
        self.train_targets = train_targets
        self.train_candidates = train_candidates
        # self.clean_mask = clean_mask
        self.train_size = len(train_users)
        self.userItemSet = userItemSet
        self.max_item_idx = max_item_idx
        self.neg_num = neg_num
        self.aug_num = aug_num
        self.input_len = len(self.train_hist_items[0])
        self.item_dist = item_dist
        self.numItem = len(item_dist)

    def __getitem__(self, index):  # index根据__len__，0~len-1
        userIdx = self.train_users[index]
        interacted_items = self.userItemSet[userIdx]  # 当前用户的items的set
        final_negs = []
        while len(final_negs) < self.neg_num:
            sampled_negs = np.random.choice(self.numItem, self.neg_num, False,
                                            self.item_dist)  # generate_itemdist里转换了概率 最后这个随机生成了一个长度为neg_num的list
            valid_negs = [x for x in sampled_negs if x not in interacted_items]  # 剔除了本来有的item
            final_negs.extend(valid_negs[:])  # 增加final_negs的长度
        # numpy.random.choice(a, size=None, replace=True, p=None)
        # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        # replace:True表示可以取相同数字，False表示不可以取相同数字
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

        final_negs = final_negs[:self.neg_num]  # 最后的negs
        return userIdx, \
               torch.tensor(self.train_hist_items[index]), \
               torch.tensor(self.train_masks[index]), \
               self.train_targets[index], \
               torch.tensor(self.train_candidates[index]), \
               torch.tensor(final_negs), \
               index

    def __len__(self):
        return self.train_size


class UnidirectTestDataset(torch.utils.data.Dataset):

    def __init__(self, test_users, test_hist_items, test_masks, test_targets):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        masks = [bs, seq_len]
        target_ids = [bs, pred_num]
        """
        self.test_users = test_users
        self.test_hist_items = test_hist_items
        self.test_masks = test_masks
        self.test_targets = test_targets
        self.test_size = len(test_users)

    def __getitem__(self, index):
        return self.test_users[index], \
               torch.tensor(self.test_hist_items[index]), \
               torch.tensor(self.test_masks[index]), \
               torch.tensor(self.test_targets[index]),

    def __len__(self):
        return self.test_size
