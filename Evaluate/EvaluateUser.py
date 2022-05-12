'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq # for retrieval topK
from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np
from time import time
#from numba import jit, autojit
# Global variables that are shared across processes
_model = None
_sess = None
_dataset = None
_K = None
_DictList = None
_gtItem = None
_user_prediction = None
_model_name = None

def init_evaluate_model(model, dataset, args):
    DictList = []

    for idx in xrange(len(dataset.testRatings)):
        user, gtItem = dataset.testRatings[idx]
        items = range(dataset.num_items) # rank on all items
        items.append(gtItem)
        user_input = np.full(len(items), user, dtype='int32')[:, None]
        item_input = np.array(items)[:,None]
        feed_dict = {model.user_input: user_input,  model.item_input: item_input}
        DictList.append(feed_dict)

    # print("already initiate the evaluate model...")
    return DictList


def gen_feed_dict(dataset):
    DictList = []
    for idx in xrange(len(dataset.testRatings)):
        user, gtItem = dataset.testRatings[idx]
        items = range(dataset.num_items) # rank on all items
        items.append(gtItem)
        user_input = np.full(len(items), user, dtype='int32')[:, None]
        item_input = np.array(items)[:,None]
        feed_dict = {'input_data/user_input:0': user_input, 
            'input_data/item_input:0': item_input}
        DictList.append(feed_dict)
    return DictList


def eval(model, sess, dataset, DictList, args, behave_type = None):
    global _model
    global _K
    global _DictList
    global _sess
    global _dataset
    global _gtItem
    global _user_prediction
    global _model_name

    _dataset = dataset
    _model = model
    _sess = sess
    _K = args.topK
    _model_name = args.model

    hits, ndcgs, _gtItem, _user_prediction = [], [], [], []


    # give predictions on users
    # for idx in xrange(len(_DictList)):
    #     if args.model == 'Multi_GMF':
    #         _gtItem.append(_dataset[0].testRatings[idx][1])
    #         _user_prediction.append(_sess.run(_model.score_buy, feed_dict = _DictList[idx]))

    #     else:
    #         _gtItem.append(_dataset.testRatings[idx][1])
    #         _user_prediction.append(_sess.run(_model.output, feed_dict = _DictList[idx]))

    # cpu_num = 4
    # pool = Pool(cpu_num)
    # res = pool.map(_eval_one_rating, range(len(_DictList)))
    # pool.close()
    # pool.join()
    # hits = [r[0] for r in res]
    # ndcgs = [r[1] for r in res]

    if args.model == 'FISM':
        for idx in xrange(len(dataset.testRatings)):
            t1 = time()
            user, gtItem = dataset.testRatings[idx]
            items = range(dataset.num_items) # rank on all items
            items.append(gtItem)
            user_input = np.full(len(items), user, dtype='int32')[:, None]
            item_input = np.array(items)[:,None]
            # item rate / item_num 
            item_rate, item_num = [], []
            item_rate_1 = dataset.trainDict[user]['buy']
            for i in items: 
                item_rate_2 = filter(lambda x:x != i, item_rate_1) 
                item_num.append(len(item_rate_2))
                item_rate_2 = item_rate_2 + [dataset.num_items]*(dataset.max_rate - len(item_rate_2))
                item_rate.append(item_rate_2)
                assert len(item_rate_2) == dataset.max_rate
            feed_dict = {model.user_input: user_input,  model.item_input: item_input,
                         model.item_rate: np.array(item_rate), model.item_num: np.array(item_num).reshape(-1, 1)}
            _feed_dict = feed_dict
            (hr, ndcg) = _eval_one_rating_FISM(idx)
            hits.append(hr)
            ndcgs.append(ndcg)

            print ('Input shape: user input %s item_input %s item_rate %s item_num %s ')
            print('user idx %d    [%.2f s]' %(idx, time()-t1))


    else:
        _DictList = DictList
        for idx in xrange(len(_DictList)):
            (hr, ndcg) = _eval_one_rating(idx, behave_type)
            hits.append(hr)
            ndcgs.append(ndcg)


    return (hits, ndcgs)

def _eval_one_rating_FISM(idx):
    gtItem = _dataset.testRatings[idx][1]
    predictions = _sess.run(_model.output, feed_dict = _feed_dict)

    rank = 0
    rank_score = predictions[gtItem]
    
    for i in predictions:
        if i > rank_score:
            rank += 1
    
    if idx < 10:
        print('idx %d\nall predictions %s \nscore %s, rank %s' % (
               idx, predictions[:5], rank_score, rank))

    if rank < _K:
        hr = 1
        ndcg = math.log(2) / math.log(rank + 2)
    else:
        hr = 0
        ndcg = 0

    return (hr, ndcg)




def _eval_one_rating(idx, behave_type):

    # predictions = _user_prediction[idx]
    # gtItem = _gtItem[idx]

    if _model_name in ['Multi_GMF', 'Multi_MLP', 'Multi_NCF']:
        gtItem = _dataset[0].testRatings[idx][1]
        predictions = _sess.run(_model.score_buy, feed_dict = _DictList[idx])

    else:
        gtItem = _dataset.testRatings[idx][1]
        if behave_type == 'ipv':
            predictions = _sess.run(_model.output1, feed_dict = _DictList[idx])
        elif behave_type == 'cart':
            predictions = _sess.run(_model.output2, feed_dict = _DictList[idx])
        elif behave_type == 'buy':
            predictions = _sess.run(_model.output3, feed_dict = _DictList[idx])
        else:
            predictions = _sess.run(_model.output, feed_dict = _DictList[idx])

    rank = 0
    rank_score = predictions[gtItem]
    
    for i in predictions:
        if i > rank_score:
            rank += 1
    
    if idx < 3:
        print('idx %d\nall predictions %s \nscore %s, rank %s' % (
               idx, predictions[:5], rank_score, rank))

    if rank < _K:
        hr = 1
        ndcg = math.log(2) / math.log(rank + 2)
    else:
        hr = 0
        ndcg = 0

    return (hr, ndcg)



