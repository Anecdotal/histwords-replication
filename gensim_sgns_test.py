import datetime
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import logging
import tempfile
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NytimesParagraphs:
    """An iterator that yields sentences (lists of str)."""

    path = ""
    slice_pct = 1.0

    def __init__(self, path, slice=1.0, paths=[]):
        self.path = path
        self.slice_pct = slice
        self.paths = paths

    def __iter__(self):

        if self.paths != []:
            for path in self.paths:

                with open(path, encoding='utf-8') as file:
                    texts = json.load(file)

                    # only keep `slice_perc` % of texts
                    if self.slice_pct != 1.0:
                        texts = texts[:int(self.slice_pct * (len(texts) + 1))]

                    for txt in texts:
                        paras = " ".join(txt['paragraphs'])
                        yield utils.simple_preprocess(paras.encode('utf-8'))
        
        else:
            with open(self.path, encoding='utf-8') as file:
                texts = json.load(file)

                # only keep `slice_perc` % of texts
                if self.slice_pct != 1.0:
                    texts = texts[:int(self.slice_pct * (len(texts) + 1))]

                for txt in texts:
                    paras = " ".join(txt['paragraphs'])
                    yield utils.simple_preprocess(paras.encode('utf-8'))


from gensim.models.callbacks import CallbackAny2Vec
class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, eps, lr):
        self.epoch = 0
        self.loss_to_be_subbed = 0
        self.total_epochs = eps
        self.lr = lr

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subbed
        self.loss_to_be_subbed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        # add final loss to grid search
        losses_by_hyp[(self.epoch, self.lr)] = loss_now
            
        self.epoch += 1


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
    in_base_embed.wv.fill_norms(force=True)
    in_other_embed.wv.fill_norms(force=True)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
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
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        #print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)

# align all time slice models to final time slice
# TODO: parameterize saving the unaligned models
def align_years(years):
    base_embed = models[years[-1]]

    for year in years[:-1]:

        year_embed = models[year]

        print("Aligning year:", year)
        aligned_embed = smart_procrustes_align_gensim(base_embed, year_embed)

        print("Writing year:", year)
        models[year + len(years)] = aligned_embed
        aligned_embed.wv.save_word2vec_format('./embeddings/nytimes-big/sgns_vectors_aligned_' + str(year) + '_all.txt', binary=False)

    year = years[-1]
    print("Writing year:", year)
    models[year + len(years)] = base_embed
    base_embed.wv.save_word2vec_format('./embeddings/nytimes-big/sgns_vectors_aligned_' + str(year) + '_all.txt', binary=False)

def yao_align_years():
    return None

def grid_search_hyperparams(year, minLR, maxLR, num_lr_steps, eps):
    path_i = "../headlines/nytimes_headlines_histwords_" + str(year) + ".json"

    # train models for all ranges, combos of LR, Epochs
    lr_step = (maxLR - minLR) / num_lr_steps
    for lr in [minLR + i*lr_step for i in range(num_lr_steps)]:
        sentences = NytimesParagraphs(path_i)

        model = gensim.models.Word2Vec(vector_size=VECTOR_SIZE, window=8, sg=USE_SKIP_GRAM, epochs=eps, alpha=lr,
                                    sentences=sentences, min_count=2, compute_loss=True,
                                    callbacks=[callback(eps, lr)])
        
    # find hyperparams with best loss
    best_eps, best_lr = min(losses_by_hyp, key=losses_by_hyp.get)
    print(losses_by_hyp)
    print("best hyperparams:", best_eps, "epochs &",  best_lr, "lr; with final loss of", losses_by_hyp[(best_eps, best_lr)])

    return best_eps, best_lr

YEARS = range(1990, 2017)

EPOCHS = 5
INIT_LR = 0.7
VECTOR_SIZE = 100
USE_SKIP_GRAM = 1

# for 50 sized vectors, best ep, lr is (18, 0.47) -- but the word vectors seem like noise
# for 100 sized vectors, best ep, lr is (16, 0.7) -- but the word vectors seem like noise
# for 16 sized vectors, best ep, lr is (16, 0.72) -- but the word vectors seem like noise

losses_by_hyp = {}

models = {}
    
#EPOCHS, INIT_LR = grid_search_hyperparams(2015, 0.002, 0.8, 20, 20)

import time
print(datetime.datetime.now().strftime('%x').replace('/', ''))

print("training whole thing...")
sentences = NytimesParagraphs("", paths=["../NYT_archive/paragraphs-" + str(year) + ".json" for year in YEARS])

model = gensim.models.Word2Vec(vector_size=VECTOR_SIZE, window=5, sg=USE_SKIP_GRAM, epochs=EPOCHS, sample=0.0001,
                                alpha=INIT_LR, min_alpha=0.07, sentences=sentences, min_count=200, compute_loss=True,
                            callbacks=[callback(EPOCHS, INIT_LR)])

date = datetime.datetime.now().strftime('%x').replace('/', '')

model.wv.save_word2vec_format('./embeddings/nytimes-big/sgns_vectors_unaligned_all_years_' + date + '.txt', binary=False)

for sl_pct in [1.0]: #float(10**x) * 1.0 for x in range(0)]:

    total_wv_start = time.time()

    train_per_year = 0.0

    for year in YEARS:

        train_start = time.time()

        path_i = "../NYT_archive/paragraphs-" + str(year) + ".json"

        sentences = NytimesParagraphs(path_i, slice=sl_pct)

        print("training", year, "...", end="", flush=True)

        model = gensim.models.Word2Vec(vector_size=VECTOR_SIZE, window=5, sg=USE_SKIP_GRAM, epochs=EPOCHS, sample=0.00001,
                                       alpha=INIT_LR, min_alpha=0.1, sentences=sentences, min_count=20, compute_loss=True,
                                    callbacks=[callback(EPOCHS, INIT_LR)])

        models[year] = model

        date = datetime.datetime.now().strftime('%x').replace('/', '')

        model.wv.save_word2vec_format('./embeddings/nytimes-big/sgns_vectors_unaligned_' + str(year) + '_' + date + '.txt', binary=False)

        train_per_year += time.time() - train_start

    print("single year avg:", train_per_year / float(len(YEARS)))

    align_years(YEARS) 

   # print("for", sl_pct, ":", time.time() - total_wv_start)
    #print("...................................")



def cos_sim(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))
'''

wv_by_year = lambda word: lambda year: models[year].wv[word]

def compare_cos(w1, w2, year):
    wv1 = wv_by_year(w1)(year)
    wv2 = wv_by_year(w2)(year)

    return cos_sim(wv1, wv2)

# pre-alignment: loop through all years, check cosine similarity of sample words across each year
cWORDS = ['president', 'senator', 'obama', 'trump', 'politics', 'mask', 'flu', 'sick', 'shutdown']

com = 'president'
cos_sims_f = {}
for year in YEARS:
    cos_sims = {}
    word = cWORDS[-1]
    for w2 in cWORDS:
        cos_sim_f = []
        for word in cWORDS:
            cos_sim_f.append(round(compare_cos(word, w2, year), 3))
        
        cos_sims[w2] = cos_sim_f

        if w2 == com:
            cos_sims_f[year] = cos_sim_f
        
        print("for", year, w2, "has similarities of:", cos_sim_f)

    print("for pairs against president:")
    for w in ['president', 'mask', 'shutdown']:
        com_sim = cos_sims[com]
        w_sim = cos_sims[w]
        sim_diffs = [round(w_sim[i] - com_sim[i], 3) for i in range(len(w_sim))]
        print("for", year, w, "vs", com, ":", sim_diffs)
    print("......")

print("between years for president:")

between_years_df = pd.DataFrame(index=YEARS, columns=YEARS)
wv_com = wv_by_year(com)

for y1 in YEARS:
    for y2 in YEARS:
        between_years_df.at[y1, y2] = round(cos_sim(wv_com(y1), wv_com(y2)), 3)

print(between_years_df.to_string)



# ALIGNMENT
align_years(YEARS) 


print("...POST ALIGNMENT...")
# then do same test post-alignment

wv_by_year2 = lambda word: lambda year: models[year + 20].wv[word]
def compare_cos2(w1, w2, year):
    wv1 = wv_by_year2(w1)(year)
    wv2 = wv_by_year2(w2)(year)

    return cos_sim(wv1, wv2)

com = 'president'
cos_sims_f = {}
for year in YEARS:
    cos_sims = {}
    word = cWORDS[-1]
    for w2 in cWORDS:
        cos_sim_f = []
        for word in cWORDS:
            cos_sim_f.append(round(compare_cos2(word, w2, year), 3))
        
        cos_sims[w2] = cos_sim_f

        if w2 == com:
            cos_sims_f[year] = cos_sim_f
        
        print("for", year, w2, "has similarities of:", cos_sim_f)

    print("for pairs against president:")
    for w in ['president', 'mask', 'shutdown']:
        com_sim = cos_sims[com]
        w_sim = cos_sims[w]
        sim_diffs = [round(w_sim[i] - com_sim[i], 3) for i in range(len(w_sim))]
        print("for", year, w, "vs", com, ":", sim_diffs)
    print("......")

print("between years for president:")
between_years_df = pd.DataFrame(index=YEARS, columns=YEARS)
wv_com = wv_by_year2(com)

for y1 in YEARS:
    for y2 in YEARS:
        between_years_df.at[y1, y2] = round(cos_sim(wv_com(y1), wv_com(y2)), 3)

print(between_years_df.to_string)

print("2023 shutdown vs. prev words:")
between_years_df = pd.DataFrame(index=cWORDS, columns=YEARS)
wv_com_last = wv_by_year2('shutdown')(YEARS[-1])

for w1 in cWORDS:
    wv_fst = wv_by_year2(w1)
    for y2 in YEARS:
        between_years_df.at[w1, y2] = round(cos_sim(wv_fst(y2), wv_com_last), 3)

print(between_years_df.to_string)

print("2023 president vs. prev words:")
between_years_df = pd.DataFrame(index=cWORDS, columns=YEARS)
wv_com_last = wv_by_year2('president')(YEARS[-1])

for w1 in cWORDS:
    wv_fst = wv_by_year2(w1)
    for y2 in YEARS:
        between_years_df.at[w1, y2] = round(cos_sim(wv_fst(y2), wv_com_last), 3)

print(between_years_df.to_string)

print("2023 mask vs. prev words:")
between_years_df = pd.DataFrame(index=cWORDS, columns=YEARS)
wv_com_last = wv_by_year2('mask')(YEARS[-1])

for w1 in cWORDS:
    wv_fst = wv_by_year2(w1)
    for y2 in YEARS:
        between_years_df.at[w1, y2] = round(cos_sim(wv_fst(y2), wv_com_last), 3)

print(between_years_df.to_string)
'''