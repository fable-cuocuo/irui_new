from data_utils.RankingEvaluator import RankingEvaluator
import torch
from torch import optim
import time
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.SeqRec import SeqRec
from model.FPMC import FPMC_encoder
from model.SASRec import SASRec_encoder
from model.Bert4Rec import BERT_encoder
from model.IOSCencoder import IOSC_encoder
from model.ComiRec import ComiRec_encoder
from model.MIND import MIND_encoder


class Trainer:
    def __init__(self, config, data_model, save_dir):
        # data_model:GraphDataCollector
        train_loader = data_model.generate_train_dataloader_unidirect()
        # valid_loader = data_model.generate_valid_dataloader_unidirect()
        test_loader = data_model.generate_test_dataloader_unidirect()

        self.item_num = data_model.numItem

        self.config = config
        self.save_dir = save_dir
        self.train_type = config['train_type']  # train/eval
        self.rec_model = config['rec_model']  # BERD
        # self.seq_model = seq_model
        self.train_loader = train_loader
        self._evaluator_1 = RankingEvaluator(test_loader)
        # self._evaluator_2 = RankingEvaluator(test_loader)
        self.item_dist = np.array(data_model.item_dist)
        self.user_dist = np.array(data_model.user_dist)
        self.train_size = len(train_loader.dataset)
        self.model_save_dir = './datasets/' + self.config['dataset'] + '/model/'
        self.model_save_path = self.model_save_dir + self.rec_model + str(self.config['dataset']) + '-'
        self.save_epochs = self.config['save_epochs']

        seq_model = self.getSeqEncoder()
        rec_model = SeqRec(config, seq_model)

        if self.train_type == 'train':
            if rec_model is not None:
                self._model_ = rec_model
                self._device = config['device']
                self._model_.double().to(self._device)
                self._optimizer_ = _get_optimizer(
                    self._model_, learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
                self.scheduler = ReduceLROnPlateau(self._optimizer_, 'max', patience=10,
                                                   factor=config['decay_factor'])
            self.high_forget_rates = self.build_high_forget_rates()
            self.low_forget_rates = self.build_low_forget_rates()
            self.weight_forget_rates = self.build_weight_forget_rates()

        elif self.train_type == 'eval':
            self._device = config['device']
            self._model_ = rec_model

    def run_co(self):
        if self.train_type == 'train':
            print('=' * 60, '\n', 'Start Training', '\n', '=' * 60, sep='')
            keep_train = True
            for epoch in range(self.config['epoch_num']):
                start = time.time()
                loss_iter = 0
                rec_loss_isc_iter = 0
                rec_loss_osc_iter = 0
                modified_target_iter = 0
                modified_hist_iter = 0
                for i, batch in enumerate(self.train_loader):
                    user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_indices = batch
                    if self.config['rec_model'] != 'IOSC':
                        loss, modified_target_num, modified_hist_num = self.train_one_batch(user_id, hist_item_ids,
                                                                                            masks,
                                                                                            pos_target,
                                                                                            train_candidates,
                                                                                            neg_targets, sample_indices,
                                                                                            epoch)
                    else:
                        loss, rec_loss_isc, rec_loss_osc, modified_target_num, modified_hist_num = self.train_one_batch(
                            user_id, hist_item_ids,
                            masks,
                            pos_target,
                            train_candidates,
                            neg_targets, sample_indices,
                            epoch)
                    loss_iter += loss.item()
                    rec_loss_isc_iter += rec_loss_isc.sum().item()
                    rec_loss_osc_iter += rec_loss_osc.sum().item()
                    modified_target_iter += modified_target_num
                    modified_hist_iter += modified_hist_num
                print(f'################## epoch {epoch} ###########################')
                print(
                    f"loss: {round(loss_iter / len(self.train_loader), 4)}, len_train_loader:{len(self.train_loader)}")
                print(f"isc_loss: {round(rec_loss_isc_iter / len(self.train_loader), 4)}")
                print(f"osc_loss: {round(rec_loss_osc_iter / len(self.train_loader), 4)}")
                print(f"modified_target in every batch:{round(modified_target_iter / len(self.train_loader), 4)}")
                print(f"modified_hist in every batch :{round(modified_hist_iter / len(self.train_loader), 4)}")
                keep_train = self.evaluate(epoch)  # 验证
                end = time.time()
                one_epoch_time = end - start
                print(f"this epoch costs {one_epoch_time} s")
                print('#########################################################')
                if epoch in self.save_epochs:
                    self.save_model(epoch)
                if not keep_train:
                    break

        elif self.train_type == 'eval':
            for epoch in self.save_epochs:
                self._model_ = self.load_model(epoch)
                self._model_.double().to(self._device)
                self._evaluator_1.evaluate(model=self._model_, train_iter=0)

    def evaluate(self, iter):
        self._model_.eval()
        keep_train, ndcg10 = self._evaluator_1.evaluate(model=self._model_, train_iter=iter)
        self.scheduler.step(ndcg10)
        return keep_train

    def train_one_batch(self, user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_idices,
                        epoch_num):
        modified_target_num = 0
        modified_hist_num = 0
        num_weight_forget = self.weight_forget_rates[epoch_num]
        if self.config['rec_model'] == 'SASRec':
            self._model_.train()
            self._optimizer_.zero_grad()

            # [bs], [bs,1]
            filter_loss, modified_index, current_pre = self._model_(user_id, hist_item_ids, masks, pos_target,
                                                                    neg_targets,
                                                                    sample_idices, num_weight_forget)
            # [bs]
            list_len = len(filter_loss)
            modified_index = modified_index.squeeze()
            num_forget = int(self.high_forget_rates[epoch_num] * list_len)
            changed_choice = self.select_invalid_loss(filter_loss, modified_index, list_len, num_forget)
            self._model_.eval()
            # warm up
            # SASRec部分结束,开始做修正，还需要记录哪些的loss高，参考berd
            with torch.no_grad():
                modified_seqs, modified_target, modified_target_num, modified_hist_num = self._model_.seqs_correction(
                    hist_item_ids, current_pre, pos_target, changed_choice,
                    train_candidates)

            self._model_.train()
            # 用bert4Rec做推荐

            recommender_output = self._model_.recommender_forward(hist_item_ids, masks)
            recommender_loss = self._model_.recommender_loss(recommender_output, pos_target, neg_targets)
            recommender_loss = recommender_loss.sum()
            total_recommender_loss = recommender_loss

            # if epoch_num >= 5:
            modified_recommender_output = self._model_.recommender_forward(modified_seqs, masks)
            modified_recommender_loss = self._model_.recommender_loss(modified_recommender_output,
                                                                      modified_target,
                                                                      neg_targets)
            modified_rec_loss = modified_recommender_loss.sum()
            total_recommender_loss += modified_rec_loss

            all_loss = total_recommender_loss + filter_loss.sum()
            all_loss.backward()
            self._optimizer_.step()
            return all_loss, modified_target_num, modified_hist_num

        if self.config['rec_model'] == 'BERT':
            self._model_.train()
            self._optimizer_.zero_grad()
            loss = self._model_(user_id, hist_item_ids, masks, pos_target, neg_targets, sample_idices)
            all_loss = loss.sum()
            all_loss.backward()
            self._optimizer_.step()
            return all_loss, modified_target_num, modified_hist_num

        if self.config['rec_model'] == 'COMIREC' or self.config['rec_model'] == 'MIND':
            self._model_.train()
            self._optimizer_.zero_grad()
            loss = self._model_(user_id, hist_item_ids, masks, pos_target, neg_targets, sample_idices)
            all_loss = loss.sum()
            all_loss.backward()
            self._optimizer_.step()
            return all_loss, modified_target_num, modified_hist_num

        if self.config['rec_model'] == 'IOSC':
            self._model_.train()
            self._optimizer_.zero_grad()
            loss_sum, rec_loss_isc, rec_loss_osc, modified_index, common_prefer, personal_prefer = self._model_(user_id,
                                                                                                                hist_item_ids,
                                                                                                                masks,
                                                                                                                pos_target,
                                                                                                                neg_targets,
                                                                                                                sample_idices,
                                                                                                                num_weight_forget)
            # loss_sum 总loss； loss1 ISCloss； loss2 OSCloss； modified_index 权重最小的index；prefer 两个prefer;
            # 对loss1和loss进行排序
            list_len = len(loss_sum)
            modified_index = modified_index.squeeze()
            high_num_forget = int(self.high_forget_rates[epoch_num] * list_len)
            low_num_forget = int(self.low_forget_rates[epoch_num] * list_len)
            changed_choice, prefer, no_loss = self.select_invalid_loss_iosc(rec_loss_isc, rec_loss_osc, common_prefer,
                                                                   personal_prefer, modified_index, list_len,
                                                                   high_num_forget, low_num_forget)
            #偏好修改的重新计算loss
            # loss = self._model_.recommender_loss(prefer, pos_target, neg_targets)
            #修正目标的不参与训练
            loss_kept_bool = np.ones(list_len)
            loss_kept_bool[no_loss] = 0
            loss_kept_bool = torch.tensor(loss_kept_bool).to(self._device)
            loss_update = loss_sum * loss_kept_bool
            # this_loss = loss_update.sum()


            self._model_.eval()
            # SASRec部分结束,开始做修正，还需要记录哪些的loss高，参考berd
            with torch.no_grad():
                # 用融合后的prefer来算相似度
                modified_seqs, modified_target, modified_target_num, modified_hist_num = self._model_.seqs_correction(
                    hist_item_ids, prefer, pos_target, changed_choice,
                    train_candidates)
            self._model_.train()
            # 用bert4Rec做推荐，完全不考虑用户偏好
            

            # if epoch_num >= 5:
            modified_recommender_output = self._model_.recommender_forward(modified_seqs, masks)
            modified_recommender_loss = self._model_.recommender_loss(modified_recommender_output,
                                                                      modified_target,
                                                                      neg_targets)

            all_loss = modified_recommender_loss.sum() + loss_update.sum()
            all_loss.backward()
            self._optimizer_.step()
            # print(f"loss_sum: {round(loss_sum / len(self.train_loader), 4)}, rec_loss_isc: {round(rec_loss_isc/ len(self.train_loader), 4)}, rec_loss_osc: {round(rec_loss_osc/ len(self.train_loader), 4)}")

            return all_loss, rec_loss_isc, rec_loss_osc, modified_target_num, modified_hist_num

    def build_high_forget_rates(self):
        forget_rates = np.ones(self.config['epoch_num']) * self.config['high_loss_drop']
        forget_rates[:20] = np.linspace(0, self.config['high_loss_drop'], 20)
        return forget_rates

    def build_low_forget_rates(self):
        forget_rates = np.ones(self.config['epoch_num']) * self.config['low_loss_drop']
        forget_rates[:20] = np.linspace(0, self.config['low_loss_drop'], 20)
        return forget_rates

    def build_weight_forget_rates(self):
        weight_forget_rates = np.ones(self.config['epoch_num']) * self.config['weight_drop']
        weight_forget_rates[:20] = np.linspace(0, 0, 20)
        return weight_forget_rates

    def select_invalid_loss(self, loss, modified_index, list_len, num_forget):
        ind_loss_sorted = np.argsort(loss.cpu().data)
        if num_forget is 0:
            changed_loss_ind = set()
        else:
            changed_loss_ind = set(ind_loss_sorted[-num_forget:].tolist())
        changed_loss_ind = list(changed_loss_ind)
        target_changed_bool = np.zeros(list_len)
        target_changed_bool[changed_loss_ind] = -2
        target_changed_bool = torch.tensor(target_changed_bool)

        if torch.cuda.is_available():
            target_changed_bool = target_changed_bool.to(torch.device('cuda'))

        changed_choice = torch.where(target_changed_bool == -2, target_changed_bool, modified_index)
        return changed_choice

    def select_invalid_loss_iosc(self, ISCloss, OSCloss, common_prefer, personal_prefer, modified_index, list_len,
                                 high_num_forget, low_num_forget):
        global common_indices, top_ISCloss_indices, top_OSCloss_indices
        ISC_ind_loss_sorted = np.argsort(ISCloss.cpu().data)
        OSC_ind_loss_sorted = np.argsort(OSCloss.cpu().data)
        target_changed_bool = np.zeros(list_len)
        changed_loss_ind = []

        if high_num_forget is 0:
            common_indices = set()
        else:
            top_ISCloss_indices = set(ISC_ind_loss_sorted[-high_num_forget:].tolist())
            top_OSCloss_indices = set(OSC_ind_loss_sorted[-high_num_forget:].tolist())
            common_indices = top_ISCloss_indices.intersection(top_OSCloss_indices)

            changed_loss_ind = list(common_indices)
            target_changed_bool[changed_loss_ind] = -2
            target_changed_bool = torch.tensor(target_changed_bool)
            # 识别loss1中位于顶部位置但不在loss2顶部中的索引
            top_loss1_exclusive_indices = top_ISCloss_indices.difference(top_OSCloss_indices)
            target_changed_bool[list(top_loss1_exclusive_indices)] = -1

        # Convert to PyTorch tensor
        target_changed_bool = torch.tensor(target_changed_bool)

        if torch.cuda.is_available():
            target_changed_bool = target_changed_bool.to(torch.device('cuda'))

        changed_choice = torch.where(target_changed_bool == -2, target_changed_bool, modified_index)
        # 因此，这段代码的作用是根据 target_changed_bool 中的布尔值选择要在 changed_choice 中保留的值，如果 target_changed_bool 中的值是 -2，则直接使用它；否则，使用 modified_index 中的相应值。

        # 融合偏好
        prefer = (personal_prefer + common_prefer) / 2
        if low_num_forget != 0:
            buttom_ISCloss_indices = set(ISC_ind_loss_sorted[:low_num_forget].tolist())
            buttom_OSCloss_indices = set(OSC_ind_loss_sorted[:low_num_forget].tolist())
            set_personal_list = list(buttom_OSCloss_indices.intersection(top_ISCloss_indices))#个性化(OSC)更适用，非个性化不适用
            set_common_list = list(buttom_ISCloss_indices.intersection(top_OSCloss_indices))#个性化适用，非个性化不适用
            prefer[set_common_list] = common_prefer[set_common_list]
            prefer[set_personal_list] = personal_prefer[set_personal_list]

        # 修改target的 loss为0
        return changed_choice, prefer, changed_loss_ind

    def save_model(self, epoch_num):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        save_path = self.model_save_path + str(epoch_num) + '-model.pkl'
        torch.save(self._model_, save_path)
        print(f'model saved at {save_path}')

    def getSeqEncoder(self):
        if self.config['rec_model'] == 'SASRec':
            return SASRec_encoder(self.config)
        elif self.config['rec_model'] == 'FPMC':
            return FPMC_encoder(self.config)
        elif self.config['rec_model'] == 'BERT':
            return BERT_encoder(self.config)
        elif self.config['rec_model'] == 'COMIREC':
            return ComiRec_encoder(self.config)
        elif self.config['rec_model'] == 'MIND':
            return MIND_encoder(self.config)
        elif self.config['rec_model'] == 'IOSC':
            return IOSC_encoder(self.config)

    def load_model(self, epoch_num):
        load_path = self.model_save_path + str(epoch_num) + '-model.pkl'
        print(f'loading model from {load_path}')
        return torch.load(load_path)


def _get_optimizer(model, learning_rate, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
