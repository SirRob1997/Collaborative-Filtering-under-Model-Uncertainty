import numpy as np
import sys
import os 
import time

import reachability as re

import pandas as pd
import scipy

import multiprocessing
from multiprocessing import Pool



def construct_params(item_factor, user_factor, df_all, user_ids, selected_items,
                      b_items, b_users, b0, user_cutoff=None):
    SEED = 38419
    n_items, _ = item_factor.shape
    n_users, _ = user_factor.shape
    pred_ratings = user_factor.dot(item_factor.T)
    user_cutoff = n_users if user_cutoff is None else user_cutoff 

    # each user, one of [n_reach_hist, n_reach_next, n_reach_random], each N
    user_hist_n = np.zeros(user_cutoff) # number of items in each users history
    # each item, one of [n_reach_hist, n_reach_next, n_reach_random], each N
    item_hist_n = np.zeros(n_items) # number of times each item is in history

    all_seen_items = []; all_ratings = []; random_recs = []
    print("Populating user history and recs")
    for u in range(user_cutoff):
        if u % 10 == 0: print("user {} of {}".format(u, user_cutoff))

        seen_items = np.array(df_all[df_all['user_id']==user_ids[u]]['artist_id_ordered'])

        ratings = np.array(df_all[df_all['user_id']==user_ids[u]]['rating'])
        if selected_items is not None:
            new_seen_items = []; new_ratings = []
            for i in range(len(seen_items)):
                new_ind = np.where(selected_items==seen_items[i])[0]
                if len(new_ind) > 0:
                    new_seen_items.append(new_ind[0]); new_ratings.append(ratings[i])
            seen_items = new_seen_items
            ratings = new_ratings
        all_seen_items.append(seen_items)
        all_ratings.append(ratings)
        user_hist_n[u] = len(seen_items)
        item_hist_n[seen_items] += 1
        pred_ratings[u, seen_items] = -np.inf
        random_recs.append(np.array([i for i in range(n_items) if i not in seen_items]))
    
    print(user_hist_n)
    np.random.seed(SEED)
    [np.random.shuffle(random_recs[u]) for u in range(user_cutoff)]
    sorted_ratings = np.argsort(pred_ratings, axis=1)
    
    return item_hist_n, user_hist_n, all_seen_items, all_ratings, sorted_ratings, random_recs

def compute_user_reach(u, seen_items, ratings, item_factor, Ns, bias, next_recs, random_recs,
                        bounds=(0,5)):
    if u % 3 == 0: print('currently on user', u)
    if len(seen_items) > 0:
        user_reachable_hist = re.get_user_aligned_reachable_top_n_items(item_factor.T, Ns, 
                               mutable_items=seen_items,
                               reg=0.04, bias=bias, constraints=bounds)
        num_hist_reachable = [len(a) for a in user_reachable_hist]
    else:
        num_hist_reachable = [0 for _ in Ns]

    user_reachable_next = []; user_reachable_random = []
    for N in Ns:
        user_reachable_next += re.get_user_aligned_reachable_top_n_items(item_factor.T, [N], 
                               immutable_items=seen_items,
                               ratings=ratings,
                               mutable_items=next_recs[-N:],
                               reg=0.04, bias=bias, constraints=bounds)

        user_reachable_random += re.get_user_aligned_reachable_top_n_items(item_factor.T, [N], 
                               immutable_items=seen_items,
                               ratings=ratings,
                               mutable_items=random_recs[:N],
                               reg=0.04, bias=bias, constraints=bounds)
    return np.array([num_hist_reachable, [len(a) for a in user_reachable_next], [len(a) for a in user_reachable_random]])

def compute_user_reach_difficulty(u, target_item, seen_items, ratings, item_factor, Ns, bias, next_recs, random_recs,
                        bounds=(0,5)):
    assert Ns == [1]
    
    if u % 3 == 0: print('currently on user', u)
    if len(seen_items) > 0:
        success, val = re.get_optimal_actions_cvx(item_factor.T, target_item, 
                        mutable_items=seen_items, immutable_items=[], ratings=ratings,
                        bias = bias,
                        l2_reg=0.04, rating_bounds=bounds)
        if not success: val = np.inf
        difficulty_hist = [val]
    else:
        difficulty_hist = [np.inf]

    difficulty_next = []; difficuty_random = []
    for N in Ns:
        success, val = re.get_optimal_actions_cvx(item_factor.T, target_item, 
                        mutable_items=next_recs[-20:], immutable_items=seen_items, ratings=ratings,
                        bias = bias,
                        l2_reg=0.04, rating_bounds=bounds)
        if not success: val = np.inf
        difficulty_next += [val]
        

        success, val = re.get_optimal_actions_cvx(item_factor.T, target_item, 
                        mutable_items=random_recs[:20], immutable_items=seen_items, ratings=ratings,
                        bias = bias,
                        l2_reg=0.04, rating_bounds=bounds)
        if not success: val = np.inf
        difficuty_random += [val]
        
    return np.array([difficulty_hist, difficulty_next, difficuty_random])

def read_model(filename, datapath):
    datafile = os.path.join(datapath, filename)
    data = np.load(datafile+'.npz', allow_pickle=True)
    
    global_bias = data['global_bias']
    weights = data['weights']
    pairwise_interactions = data['pairwise_interactions']

    groups = np.array(pd.read_csv(datapath+'meta_1.text', header=None)).flatten()
    b0 = global_bias
    b_items = weights[groups==1] if weights.size > 0 else np.zeros(sum(groups==1))
    b_users = weights[groups==0] if weights.size > 0 else np.zeros(sum(groups==0))
    item_factor = pairwise_interactions[groups==1]
    user_factor = pairwise_interactions[groups==0]
    return (b0, b_items, b_users, item_factor, user_factor), data['preds']

if __name__ == "__main__":
    filename = sys.argv[1]
    latent_dim = int(sys.argv[2])

    ## Dataset Dependent Parameters

    if filename == 'ml':
        datapath = './ml-10M100K/' 
        tag = ''
        test_df = pd.read_csv(datapath+'r1.test', sep='::', header=None, names=['user_id','movie_id','rating'], usecols=[0,1,2])
        dfm = pd.read_csv(datapath+'movie_genres_stats.csv')

    elif filename == 'fm':
        datapath = './lastfm-dataset-1K/'
        tag =  '_nb_r=0.08_ss=0.001'
        test_df = pd.read_csv(datapath+'lfm1k-play-count.test', sep=',', header=None, names=['user_id','artist_id','rating'], usecols=[0,1,2])
        dfm = pd.read_csv(datapath+'artist_genres_stats.csv')

    testing_cutoff = 1000
    USER_LIMIT = 100
    Ns = [1,2,3,5,20,100]

    ## Loading Model Data

    model, preds = read_model(filename=filename+'_res_k={}{}'.format(latent_dim,tag), datapath=datapath)
    b0, b_items, b_users, item_factor, user_factor = model

    ## Computing RMSE

    print(test_df['rating'].max(), test_df['rating'].min())
    errs = test_df['rating'] - preds
    RMSE = np.sqrt(np.mean(np.power(errs,2)))
    print('RMSE:',RMSE)

    ## Aligned-Item Reachability

    print('Computing item reachability...', end=' ')
    if item_factor.shape[0] > 100000:
        item_factor = item_factor[:100000]
        b_items = b_items[:100000]
        print(item_factor.shape)
    reachable_items = re.get_latent_aligned_reachable_top_n_items(item_factor.T.astype(np.float32), Ns, item_bias=b_items)
    print([len(r) for r in reachable_items])

    ## Loading test data for user history construction

    print('Loading raw indexed data')
    df = {}
    for s in ['test', 'train']:
        df[s] = pd.read_csv(datapath+'r1_indexed.'+s, usecols=[1,2,3,4])
    df_all = pd.concat([df['test'],df['train']])

    indices = pd.read_csv(datapath+'index_map_1.text', header=None, sep=' ')
    user_ids = np.array(indices[indices[0]==0][1])

    ## Pruning number of users/items

    SEED = 2453
    np.random.seed(SEED)

    ordered_movies = np.argsort(dfm['n_ratings_test']+dfm['n_ratings_train'])
    ordered_inds = np.array(dfm['ordered_id'].reindex(ordered_movies))
    selected_items = ordered_inds[-testing_cutoff:]
    item_factor = item_factor[selected_items]
    b_items = b_items[selected_items]
    if testing_cutoff < user_factor.shape[0]:
        selected_users = np.random.choice(user_factor.shape[0], size=testing_cutoff, replace=False)
        user_factor = user_factor[selected_users]
        b_users = b_users[selected_users]
        user_ids = user_ids[selected_users]
    
    ## User-based reachability 
  
    print('computing user reachability...')

    res = construct_params(item_factor, user_factor, df_all, user_ids, selected_items,
                              b_items, b_users, b0, user_cutoff=USER_LIMIT)
    item_hist_n, user_hist_n, all_seen_items, all_ratings, sorted_ratings, random_recs = res
    
    args = [(u, all_seen_items[u], all_ratings[u], 
            item_factor, Ns, (b_items, b_users[u], b0), 
            sorted_ratings[u], random_recs[u].flatten()) for u in range(USER_LIMIT)]
        
    print('parallel processing user reachability')
    pool = Pool(processes=multiprocessing.cpu_count()-2)
    res = pool.starmap(compute_user_reach, args) 
        

    aligned_user_reach = np.array(res)
        

    ## Saving feasibility info

    savefile = os.path.join(datapath, filename+'_test_reachability_k={}.npz'.format(latent_dim))
    print('saving to', savefile)
    np.savez(savefile, Ns=Ns, RMSE=RMSE, reachable_items=reachable_items, aligned_user_reach=aligned_user_reach,
                user_hist_n=user_hist_n, item_hist_n=item_hist_n)


    ## Running difficulty analysis

    Ns = [1]
    target_item = 500

    args = []; ret_user_factor = []; ret_user_hist_n = []
    for u in range(USER_LIMIT):
        arg = (u, target_item, all_seen_items[u], all_ratings[u], item_factor, Ns, (b_items, b_users[u], b0),
               sorted_ratings[u], random_recs[u].flatten()) 
        if target_item not in all_seen_items[u]: 
            args.append(arg)
            ret_user_factor.append(user_factor[u])
            ret_user_hist_n.append(user_hist_n[u])
    print('parallel processing user reachability', len(args))
    pool = Pool(processes=multiprocessing.cpu_count()-2)
    res = pool.starmap(compute_user_reach_difficulty, args) 
    
    user_difficulty = np.array(res)
    
    savefile = os.path.join(datapath, filename+'_test_reachability_difficulty_k={}.npz'.format(latent_dim))
    print('saving to', savefile)
    np.savez(savefile, user_difficulty=user_difficulty, target_item_factor=item_factor[target_item], user_factor=ret_user_factor,
            user_hist_n=ret_user_hist_n)
