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
def eval_analogy(models_path, w1, y1, w2, y2, top_knn=1):
    # top_knn chooses how many nearest neighbors to test against. 
    # Original model only tested closest neighbor but given noisiness of word vectors this seems misguided.

    # load models
    mod_y1 = KeyedVectors.load_word2vec_format(models_path + 'sgns_vectors_unaligned_' + str(y1) + '.txt', binary=False)
    mod_y2 = KeyedVectors.load_word2vec_format(models_path + 'sgns_vectors_unaligned_' + str(y2) + '.txt', binary=False)

    # align mod_y2 with y1
    mod_y2_aligned = smart_procrustes_align_gensim(mod_y1, mod_y2)

    # lookup word in y1
    wv1 = mod_y1[w1]

    # find nearest neighbor in y2
    preds = mod_y2_aligned.most_similar(wv1, topn=top_knn)

    w2_preds = [wrd for wrd, _ in preds]

    print(w1, "vs.", w2_preds)
    
    return w2 in w2_preds

def cos_sim(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

def eval_static(model, w1, w2, top_knn=10):
    # top_knn chooses how many nearest neighbors to test against. 
    # Original model only tested closest neighbor but given noisiness of word vectors this seems misguided.

    # lookup word in y1
    if not model.__contains__(w1) or not model.__contains__(w2):
        return False
    
    wv1 = model.get_vector(w1)

    # print distance between two words
    print(w1, w2, ":", cos_sim(wv1, model.get_vector(w2)))

    # find nearest neighbor in y2
    preds = model.most_similar(wv1, topn=top_knn)

    w2_preds = [wrd for wrd, _ in preds]

    print("top-1:", w2_preds[0], ":", cos_sim(wv1, model.get_vector(w2_preds[0])))

    print(w1, w2, "vs.", w2_preds)
    
    return w2 in w2_preds

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

#### TEST ANALOGIES ###
def eval_yao_analogies(analogies, models_path, model_years, top_knn=1, isStatic=False):
    acc_emb = 0.0
    acc_base = 0.0

    # shuffle analogies
    random.shuffle(analogies)

    for an1, an2 in analogies:

        w1, y1 = an1.split('-')
        w2, y2 = an2.split('-')

        y1, y2 = int(y1), int(y2)

        if y1 in model_years and y2 in model_years:

            eval = eval_static(KeyedVectors.load_word2vec_format(models_path, binary=False), w1, w2, top_knn) if isStatic else \
                eval_analogy(models_path, w1, y1, w2, y2, top_knn=top_knn)

            if eval:
                acc_emb += 1
                print("correct on", w1, w2, y1)

            if eval_baseline(w1, w2):
                acc_base += 1

    acc_base /= len(analogies)
    acc_emb /= len(analogies)
    print("overall accuracy:")
    print("embeddings:", acc_emb)
    print("baseline:", acc_base)

def eval_szymanski_analogies(analogies_dict, models_path, model_years, top_knn=1, isStatic=False):

    # ...
    years = analogies_dict["Year"]

    res_dict = {"total": 0, "final": (0.0, 0.0)}

    for category in analogies_dict.keys()[1:2]:

        category_res = [0.0, 0.0, 0]
        for y1 in range(len(years)):

            for y2 in range(y1 + 1, len(years)):

                if years[y1] in model_years and years[y2] in model_years:

                    w1 = analogies_dict[category][y1]
                    w2 = analogies_dict[category][y2]
                    category_res[2] += 1

                    eval = eval_static(KeyedVectors.load_word2vec_format(models_path, binary=False), w1, w2, top_knn) if isStatic else \
                        eval_analogy(models_path, w1, years[y1], w2, years[y2], top_knn=top_knn)

                    if eval:
                        category_res[0] += 1

                    if eval_baseline(w1, w2):
                        category_res[1] += 1
    
        res_dict[category] = category_res[0] / category_res[2], category_res[1] / category_res[2]

        res_dict['total'] += category_res[2]

        res_dict['final'] = res_dict['final'][0] + category_res[0], res_dict['final'][1] + category_res[1] 

    total_an = res_dict['total']
    res_dict['final'] = res_dict['final'][0] / total_an, res_dict['final'][1] / total_an

    print(res_dict)
    return res_dict

MODELS_PATH = './embeddings/nytimes-big/'
STATIC_MPATH = './embeddings/nytimes-big/sgns_vectors_unaligned_all_years_041724.txt'
ANALOGIES_PATH = './testset_1.csv'
SYZ_ANALOGIES_PATH = './szymanski_analogies.csv'
MODEL_YEARS = range(1990, 2017)

TOP_KNN = 10

analogies = load_analogies(ANALOGIES_PATH)

an_dict = pd.read_csv(SYZ_ANALOGIES_PATH)

#eval_szymanski_analogies(an_dict, MODELS_PATH, MODEL_YEARS, TOP_KNN)


eval_yao_analogies(analogies, STATIC_MPATH, MODEL_YEARS, TOP_KNN, isStatic=True)

