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

def init_evaluate_model(model,  dataset):
    DictList = []
    for idx in xrange(len(dataset.testRatings)):
        rating = dataset.testRatings[idx]
        if model.fix == 'fixed':
            items = dataset.testNegatives[idx]
        else:
            items = range(dataset.num_items)
            for i in items[::-1]:
                if not pos_samples:
                    break;
                elif pos_samples[-1] == i:
                    del items[i]
                    pos_samples.pop()
                if i == gtItem:
                    del items[i]
        items.append(gtItem)
        user = dataset.trainList[idx]
        num_idx_ = len(user)
        gtItem = rating[1]
        # Get prediction scores
        num_idx = np.full(len(items),num_idx_, dtype=np.int32 )[:,None]
        labels = np.zeros(len(items))[:, None]
        labels[-1] = 1
        user_input = []
        for i in range(len(items)):
            user_input.append(user)
        user_input = np.array(user_input)
        item_input = np.array(items)[:,None]
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx, model.item_input: item_input, model.labels: labels}
        DictList.append(feed_dict)
    print("already load the evaluate model...")
    return DictList

def eval(model, sess, dataset, DictList):
    global _model
    global _K
    global _DictList
    global _sess
    global _dataset
    _dataset = dataset
    _model = model
    _DictList = DictList
    _sess = sess
    _K = 10

    hits, ndcgs, losses = [],[],[]
    # pool = Pool(cpu_count())
    # res = pool.map(_eval_one_rating, range(len(_DictList)))
    # pool.close()
    # pool.join()
    # hits = [r[0] for r in res]
    # ndcgs = [r[1] for r in res]
    # losses = [r[2] for r in res]
    # Single thread
    # else:
    for idx in xrange(len(_DictList)):
        (hr,ndcg, loss) = _eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
        losses.append(loss)
    return (hits, ndcgs, losses)

def _eval_one_rating(idx):
    map_item_score = {}
    items = _dataset.testNegatives[idx]  #have been appended
    gtItem = items[-1]
    predictions,loss = _sess.run([_model.output, _model.loss], feed_dict = _DictList[idx])

    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]

    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = _getHitRatio(ranklist, gtItem)
    ndcg = _getNDCG(ranklist, gtItem)
    return (hr, ndcg, loss)

def _getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def _getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
