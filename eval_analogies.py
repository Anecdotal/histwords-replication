from gensim.models import KeyedVectors
import csv
import numpy as np
from numpy.linalg import norm
import pandas as pd
import random
# NB: should come from saved unaligned year-wise models


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # re-filling the normed vectors
    in_base_embed.fill_norms(force=True)
    in_other_embed.fill_norms(force=True)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.get_normed_vectors()
    other_vecs = in_other_embed.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.vectors = (other_embed.vectors).dot(ortho)    
    
    return other_embed

def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.index_to_key)
    vocab_m2 = set(m2.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.get_vecattr(w, "count") + m2.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.key_to_index[w] for w in common_vocab]
        old_arr = m.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.key_to_index = new_key_to_index
        m.index_to_key = new_index_to_key
        
        #print(len(m.key_to_index), len(m.vectors))
        
    return (m1,m2)


#### ANALOGY EVAL ####
def eval_analogy(mod_y1, mod_y2_aligned, w1, w2, top_knn=1, freq_thresh=0.0, szymanski=False):
    # top_knn chooses how many nearest neighbors to test against. 
    # freq_thresh sets a threshold for when not to consider low frequency words: 
    #   if top_pct=0.1, then words at or below the 10th frequency percentile in 
    #   the corpus won't be considered for the analogy.
    # szymanski: if True, find most similar vectors to w1 in mod_y2_aligned rather 
    #   than mod_y1. ...Pretty sure syzmanski was just coding it wrong.

    # make sure words are in the model
    if not mod_y1.__contains__(w1) or not mod_y2_aligned.__contains__(w1):
        return False, 1.0

    # 

    # lookup word in y1
    wv1 = mod_y1[w1] if not szymanski else mod_y2_aligned[w1] 

    # print distance between two words
    wv_sim = cos_sim(wv1, mod_y2_aligned.get_vector(w2)) if mod_y2_aligned.__contains__(w2) else 0.0
    #print(w1, w2, ":", wv_sim)

    # if freq_thresh != 0.0: 
    #   slice mod_y2_aligned so that it doesn't have bottom x%

    # find nearest neighbor in y2
    preds = mod_y2_aligned.most_similar(wv1, topn=top_knn)

    w2_preds = [wrd for wrd, _ in preds]


    #print(w1, "vs.", w2_preds)

    # print frequency of w1, w2, preds
    y2_vocab_len = len(mod_y2_aligned)
    w2_idx = mod_y2_aligned.key_to_index[w2]

    y1_vocab_len = len(mod_y1)
    w1_idx = mod_y1.key_to_index[w1]

    print("w1:", w1_idx, "/", y1_vocab_len)
    print("w2:", w2_idx, "/", y2_vocab_len, "& count:", mod_y2_aligned.expandos['count'][w2])

    preds_idx_ct = [(mod_y2_aligned.key_to_index[pred], mod_y2_aligned.expandos['count'][pred]) for pred in w2_preds]

    for idx, count in preds_idx_ct:
        print("pred:", idx, "/", y2_vocab_len, "& count:", count)
    
    return w2 in w2_preds, wv_sim

def cos_sim(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

def eval_static(model, w1, w2, top_knn=10):
    # top_knn chooses how many nearest neighbors to test against. 
    # Original model only tested closest neighbor but given noisiness of word vectors this seems misguided.

    # make sure words are in the model
    if not model.__contains__(w1) or not model.__contains__(w2) or model.key_to_index[w1] >= len(model.vectors) or model.key_to_index[w2] >= len(model.vectors):
        return False, 1.0

    # lookup word in y1
    wv1 = model.get_vector(w1)

    # print(w1, w2, model.key_to_index[w2], len(model.vectors))

    # print distance between two words
    wv_sim = cos_sim(wv1, model.get_vector(w2))
    # print(w1, w2, ":", wv_sim)

    # find nearest neighbor in y2
    preds = []
    i = 1
    while len(preds) < top_knn and i < 100:

        pred = model.most_similar(wv1, topn=i)

        if model.key_to_index[pred[-1][0]] / len(model) < 0.2:
            preds.append(pred[-1])

        i += 1


    w2_preds = [wrd for wrd, _ in preds]

    #print("top-1:", w2_preds[0], ":", cos_sim(wv1, model.get_vector(w2_preds[0])))

    # print frequency of w1, w2, preds
    # vocab_len = len(model)
    # w2_idx = model.key_to_index[w2]

    # w1_idx = model.key_to_index[w1]

    # print("w1:", w1, round(w1_idx / vocab_len, 3), "%")
    # print("w2:", w2, round(w2_idx / vocab_len, 3), "% & count:", model.expandos['count'][w2_idx])

    # preds_idx = [(pred, model.key_to_index[pred]) for pred in w2_preds]

    # for pred, idx in preds_idx:
    #     print("pred:", pred, round(idx / vocab_len, 3), "% & count:", model.expandos['count'][idx])

    #print(w1, w2, "vs.", w2_preds)
    
    return w2 in w2_preds, wv_sim

def eval_baseline(w1, w2):
    return w1 == w2

#### READ IN TEMPORAL ANALOGIES ####
def load_analogies(a_path):
    analogies = []

    with open(a_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            analogies.append(row)

    return analogies

def load_models(models_path, static_path, model_years):

    models = {}

    print("static", end="...", flush=True)
    
    static_model = KeyedVectors.load_word2vec_format(static_path, binary=False)

    # for yr2 in model_years:
    #     print(yr2, end="...", flush=True)
    #     models[yr2] = KeyedVectors.load_word2vec_format(models_path + 'sgns_vectors_unaligned_' + str(yr2) + '_041724.txt', binary=False)

    #     # align mod_y2 with y1
    #     for yr1 in range(model_years[0],yr2):

    #         mod_y1 = models[yr1]

    #         models[(yr1, yr2)] = smart_procrustes_align_gensim(mod_y1, models[yr2])

    return models, static_model


#### TEST ANALOGIES ###
def eval_yao_analogies(analogies, gpt_results, models, static_model, model_years, top_knn=1, freq_thresh=0.0):
    acc_emb = 0.0
    acc_stat = 0.0
    acc_gpt = 0.0
    acc_base = 0.0

    wv_sim_avg = 0.0
    wv_sim_st_avg = 0.0
    total_an = 0

    # slice static model so that it doesn't have bottom freq_thresh %
    if freq_thresh > 0.0 and static_model != None: 
        print("slicing", int(len(static_model) * (1 - freq_thresh)))
        static_model.vectors = static_model.vectors[:int(len(static_model) * (1 - freq_thresh)),:]

    # shuffle analogies
    # random.shuffle(analogies)

    for i, (an1, an2) in enumerate(analogies):

        w1, y1 = an1.split('-')
        w2, y2 = an2.split('-')

        # swap analogy so y1 < y2
        if y2 < y1:
            w3, y3 = w1, y1
            w1, y1 = w2, y2
            w2, y2 = w3, y3
            

        y1, y2 = int(y1), int(y2)

        if y1 in model_years and y2 in model_years:

            total_an += 1

            # aligned test
            if y1 in models and (y1, y2) in models:
                eval, wv_sim = eval_analogy(models[y1], models[(y1, y2)], w1, w2, top_knn=top_knn, freq_thresh=freq_thresh)

                wv_sim_avg += wv_sim
            
                if eval:
                    acc_emb += 1
                    #print("correct on", w1, w2, y1)

            # static embedding test
            if static_model != None:

                eval_st, wv_sim_st = eval_static(static_model, w1, w2, top_knn=top_knn)

                wv_sim_st_avg += wv_sim_st

                if eval_st:
                    acc_stat += 1

            # gpt-4 results 
            if gpt_results != []:
                
                # print(w1, w2, y1, y2)
                # print("gpt:", gpt_results[i])

                w1_gpt, _, w2_gpt, _, gpt_res = list(map(lambda x: x.strip(), gpt_results[i]))
                
                if w1 != w1_gpt or w2 != w2_gpt:
                    print("GPT ERROR ON", w1, w2, w1_gpt, w2_gpt)

                # TODO: fix switch around on yao analogies
                if w1 in list(map(lambda r: r.lower().strip(), gpt_res.split(';'))):
                    #print("GPT correct on", w1, w2, y1, y2)
                    acc_gpt += 1

            # baseline test
            if eval_baseline(w1, w2):
                acc_base += 1

    acc_base /= total_an
    acc_emb /= total_an
    acc_stat /= total_an
    acc_gpt /= total_an
    wv_sim_avg /= total_an
    print("avg aligned sim:", wv_sim_avg / total_an)
    print("avg static sim:", wv_sim_st_avg / total_an)
    print("overall accuracy:")
    print("aligned:", acc_emb)
    print("static:", acc_stat)
    print("baseline:", acc_base)
    print("gpt-4:", acc_gpt)

def eval_szymanski_analogies(analogies_dict, models, static_model, model_years, top_knn=1, freq_thresh=0.0):

    years = analogies_dict["Year"]

    res_dict = {"total": 0, "final": (0.0, 0.0, 0.0)}

    wv_sim_avg = 0.0
    wv_sim_st_avg = 0.0


    # slice static model so that it doesn't have bottom freq_thresh %
    if freq_thresh > 0.0 and static_model != None: 
        print("slicing", int(len(static_model) * (1 - freq_thresh)))
        static_model.vectors = static_model.vectors[:int(len(static_model) * (1 - freq_thresh)),:]

    for category in analogies_dict.keys()[1:]:

        category_res = [0.0, 0.0, 0.0]
        cat_total = 0
        for y1 in range(len(years)):

            for y2 in range(y1 + 1, len(years)):

                year1, year2 = years[y1], years[y2]

                if year1 in model_years and year2 in model_years:

                    w1 = analogies_dict[category][y1]
                    w2 = analogies_dict[category][y2]
                    cat_total += 1

                    # aligned embedding test
                    if year1 in models and (year1, year2) in models:
                        eval, wv_sim = eval_analogy(models[year1], models[(year1, year2)], w1, w2, top_knn=top_knn, freq_thresh=freq_thresh)

                        wv_sim_avg += wv_sim

                        if eval:
                            category_res[0] += 1

                    # static embedding test
                    if static_model != None:

                        eval_st, wv_sim_st = eval_static(static_model, w1, w2, top_knn=top_knn)

                        wv_sim_st_avg += wv_sim_st

                        if eval_st:
                            category_res[1] += 1
                    
                    # baseline test
                    if eval_baseline(w1, w2):
                        category_res[2] += 1
    
        res_dict[category] = category_res[0] / cat_total, \
            category_res[1] / cat_total, \
            category_res[2] / cat_total

        res_dict['total'] += cat_total

        res_dict['final'] = res_dict['final'][0] + category_res[0], \
            res_dict['final'][1] + category_res[1], \
            res_dict['final'][2] + category_res[2] 

    total_an = res_dict['total']
    res_dict['final'] = res_dict['final'][0] / total_an, \
        res_dict['final'][1] / total_an, \
        res_dict['final'][2] / total_an

    print("avg aligned sim:", wv_sim_avg / total_an)
    print("avg static sim:", wv_sim_st_avg / total_an)
    print(res_dict)
    return res_dict

MODELS_PATH = './embeddings/nytimes-big/'
STATIC_MPATH = './embeddings/nytimes-big/sgns_vectors_unaligned_all_years_042424.txt'
YAO_TS1_PATH = './testset_1.csv'
YAO_TS2_PATH = './testset_2.csv'
SYZ_ANALOGIES_PATH = './szymanski_analogies.csv'

GPT_YAO_1_PATH = 'gpt_responses_yao_analogies_1.csv'
GPT_YAO_2_PATH = 'gpt_responses_yao_analogies_2.csv'

MODEL_YEARS = range(1990, 2017)

TOP_KNN = 10
FREQ_THRESH = 0.0

ts1_analogies = load_analogies(YAO_TS1_PATH)
ts2_analogies = load_analogies(YAO_TS2_PATH)

an_dict = pd.read_csv(SYZ_ANALOGIES_PATH)

gpt_results_1 = load_analogies(GPT_YAO_1_PATH)
gpt_results_2 = load_analogies(GPT_YAO_2_PATH)

print("loading models......", end="", flush=True)
models, static_model = load_models(MODELS_PATH, STATIC_MPATH, MODEL_YEARS)
print("done loading models!")

print(".......szymanski test...........")
#eval_szymanski_analogies(an_dict, models, static_model, MODEL_YEARS, TOP_KNN, FREQ_THRESH)

print(".......yao testset 1............")
eval_yao_analogies(ts1_analogies, gpt_results_1, models, static_model, MODEL_YEARS, TOP_KNN, FREQ_THRESH)

print(".......yao testset 2............")
#eval_yao_analogies(ts2_analogies, gpt_results_2, models, static_model, MODEL_YEARS, TOP_KNN, FREQ_THRESH)
