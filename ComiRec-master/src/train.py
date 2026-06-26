#coding:utf-8
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

import faiss
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
from data_iterator import DataIterator
from model import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao | custom')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='none', help='DNN | GRU4REC | ..')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--data_dir', type=str, default='', help='custom dataset directory')
parser.add_argument('--item_count', type=int, default=0, help='item vocabulary size for custom dataset')
parser.add_argument('--batch_size', type=int, default=0, help='override default batch size')
parser.add_argument('--maxlen', type=int, default=0, help='override default max sequence length')
parser.add_argument('--test_iter', type=int, default=0, help='override default eval interval')
parser.add_argument('--exp_name', type=str, default='', help='non-interactive experiment name')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing experiment directory')
parser.add_argument('--file_prefix', type=str, default='', help='train/valid/test/cate filename prefix')

best_metric = 0

def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def evaluate_full(sess, test_data, model, model_path, batch_size, item_cate_map, save=True, coef=None):
    topN = args.topN

    item_embs = model.output_item(sess)

    try:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        index.add(item_embs)
    except Exception as e:
        # fallback to CPU faiss when GPU index is unavailable
        index = faiss.IndexFlatIP(args.embedding_dim)
        index.add(item_embs)

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0
    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

        user_embs = model.output_user(sess, [hist_item, hist_mask])

        if len(user_embs.shape) == 2:
            D, I = index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(I[i]):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []
                if coef is None:
                    item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    item_list.sort(key=lambda x:x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            item_cor_list.append(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    origin_item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                    origin_item_list.sort(key=lambda x:x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        max_index = 0
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:
                                break
                        item_list_set.add(item_list[max_index][0])
                        item_cor_list.append(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        
        total += len(item_id)
    
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}

def get_model(dataset, model_type, item_count, batch_size, maxlen):
    if model_type == 'DNN': 
        model = Model_DNN(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'GRU4REC': 
        model = Model_GRU4REC(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = Model_MIND(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    return model

def _open_train_log(exp_name):
    log_dir = os.path.join('runs', exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train.log')
    log_file = open(log_path, 'a', encoding='utf-8')
    log_file.write(f'\n=== training started {datetime.now(timezone.utc).isoformat()} ===\n')
    log_file.flush()
    return log_file, log_path


def _log_line(log_file, msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()


def _save_results(exp_name, payload):
    results_path = os.path.join('runs', exp_name, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return results_path


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    para_name = '_'.join([dataset, model_type, 'b'+str(batch_size), 'lr'+str(lr), 'd'+str(args.embedding_dim), 'len'+str(maxlen)])
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = para_name + '_' + str(int(time.time()))

    if os.path.exists('runs/' + exp_name) and save:
        if args.overwrite:
            shutil.rmtree('runs/' + exp_name)
        else:
            raise ValueError('Experiment exists: %s (use --overwrite or another --exp_name)' % exp_name)

    return exp_name

def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        test_iter = 50,
        model_type = 'DNN',
        lr = 0.001,
        max_iter = 100,
        patience = 20
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)

    best_model_path = "best_model/" + exp_name + '/'

    gpu_options = tf.GPUOptions(allow_growth=True)

    writer = SummaryWriter('runs/' + exp_name)
    log_file, log_path = _open_train_log(exp_name)
    history = []

    item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
        valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)
        
        model = get_model(dataset, model_type, item_count, batch_size, maxlen)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        _log_line(log_file, 'training begin')
        _log_line(
            log_file,
            'config: model=%s dataset=%s batch=%d maxlen=%d lr=%s max_iter=%d topN=%d'
            % (model_type, dataset, batch_size, maxlen, lr, max_iter, args.topN),
        )

        start_time = time.time()
        iter = 0
        try:
            loss_sum = 0.0
            trials = 0

            for src, tgt in train_data:
                data_iter = prepare_data(src, tgt)
                loss = model.train(sess, list(data_iter) + [lr])
                
                loss_sum += loss
                iter += 1

                if iter % test_iter == 0:
                    metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, item_cate_map)
                    avg_loss = loss_sum / test_iter
                    log_str = 'iter: %d, train loss: %.4f' % (iter, avg_loss)
                    if metrics != {}:
                        log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                    _log_line(log_file, exp_name)
                    _log_line(log_file, log_str)

                    writer.add_scalar('train/loss', avg_loss, iter)
                    if metrics != {}:
                        for key, value in metrics.items():
                            writer.add_scalar('eval/' + key, value, iter)
                        history.append({'iter': iter, 'train_loss': avg_loss, **metrics})
                    
                    if 'recall' in metrics:
                        recall = metrics['recall']
                        global best_metric
                        if recall > best_metric:
                            best_metric = recall
                            model.save(sess, best_model_path)
                            trials = 0
                        else:
                            trials += 1
                            if trials > patience:
                                break

                    loss_sum = 0.0
                    test_time = time.time()
                    _log_line(log_file, "time interval: %.4f min" % ((test_time-start_time)/60.0))
                
                if iter >= max_iter * 1000:
                        break
        except KeyboardInterrupt:
            _log_line(log_file, '-' * 89)
            _log_line(log_file, 'Exiting from training early')

        model.restore(sess, best_model_path)

        valid_metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, item_cate_map, save=False)
        _log_line(log_file, ', '.join(['valid ' + key + ': %.6f' % value for key, value in valid_metrics.items()]))

        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        test_metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, item_cate_map, save=False)
        _log_line(log_file, ', '.join(['test ' + key + ': %.6f' % value for key, value in test_metrics.items()]))

        results_path = _save_results(
            exp_name,
            {
                'exp_name': exp_name,
                'model_type': model_type,
                'dataset': dataset,
                'topN': args.topN,
                'best_valid_recall': best_metric,
                'valid': valid_metrics,
                'test': test_metrics,
                'total_iters': iter,
                'train_duration_min': (time.time() - start_time) / 60.0,
                'history': history,
                'finished_at': datetime.now(timezone.utc).isoformat(),
            },
        )
        _log_line(log_file, 'results saved to %s' % results_path)
        log_file.close()

def test(
        test_file,
        cate_file,
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)
    item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, item_cate_map, save=False, coef=args.coef)
        print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))

def output(
        item_count,
        dataset = "book",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DNN',
        lr = 0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, batch_size, maxlen)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('output/' + exp_name + '_emb.npy', item_embs)

if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    tf.compat.v1.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    elif args.dataset == 'custom':
        if not args.data_dir:
            raise ValueError('--data_dir is required when --dataset=custom')
        if args.item_count <= 0:
            raise ValueError('--item_count must be > 0 when --dataset=custom')
        path = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
        item_count = args.item_count
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    else:
        raise ValueError('Unsupported dataset: %s' % args.dataset)

    if args.batch_size > 0:
        batch_size = args.batch_size
    if args.maxlen > 0:
        maxlen = args.maxlen
    if args.test_iter > 0:
        test_iter = args.test_iter
    
    file_prefix = args.file_prefix if args.file_prefix else args.dataset
    train_file = path + file_prefix + '_train.txt'
    valid_file = path + file_prefix + '_valid.txt'
    test_file = path + file_prefix + '_test.txt'
    cate_file = path + file_prefix + '_item_cate.txt'
    dataset = args.dataset

    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file, 
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter, 
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience)
    elif args.p == 'test':
        test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size, 
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, 
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')
