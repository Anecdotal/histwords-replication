from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import logging
import tempfile
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

cYEARS = range(2013, 2024)

class Headlines:
    """An iterator that yields sentences (lists of str)."""

    path = "nytimes_headlines_histwords_2013.json"

    def __init__(self, path):
        self.path = path

    def __iter__(self):

        with open(self.path) as file:
            articles = json.load(file)

            for ar in articles:
                yield utils.simple_preprocess(ar['headline'].encode('utf-8'))


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
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)

def align_years(years):
    first_iter = True
    base_embed = None

    for year in years:

        year_embed = models[year]

        print("Aligning year:", year)
        if first_iter:
            aligned_embed = year_embed
            first_iter = False
        else:
            aligned_embed = smart_procrustes_align_gensim(base_embed, year_embed)
        base_embed = aligned_embed

        print("Writing year:", year)
        models[year + 20] = aligned_embed
        aligned_embed.wv.save_word2vec_format('./embeddings/nytimes/sgns_vectors_' + str(year) + '.txt', binary=False)

EPOCHS = 16
INIT_LR = 0.72
VECTOR_SIZE = 16

# for 50 sized vectors, best ep, lr is (18, 0.47) -- but the word vectors seem like noise
# for 100 sized vectors, best ep, lr is (16, 0.7) -- but the word vectors seem like noise
# for 16 sized vectors, best ep, lr is (16, 0.72) -- but the word vectors seem like noise

losses_by_hyp = {}

models = {}

def grid_search_hyperparams(year, minLR, maxLR, num_lr_steps, eps):
    path_i = "./headlines/nytimes_headlines_histwords_" + str(year) + ".json"

    # train models for all ranges, combos of LR, Epochs
    lr_step = (maxLR - minLR) / num_lr_steps
    for lr in [minLR + i*lr_step for i in range(num_lr_steps)]:
        sentences = Headlines(path_i)

        model = gensim.models.Word2Vec(vector_size=VECTOR_SIZE, epochs=eps, alpha=lr,
                                    sentences=sentences, min_count=2, compute_loss=True, sg=False,
                                    callbacks=[callback(eps, lr)])
        
    # find hyperparams with best loss
    best_eps, best_lr = min(losses_by_hyp, key=losses_by_hyp.get)
    print(losses_by_hyp)
    print("best hyperparams:", best_eps, "epochs &",  best_lr, "lr; with final loss of", losses_by_hyp[(best_eps, best_lr)])

    return best_eps, best_lr
    
#EPOCHS, INIT_LR = grid_search_hyperparams(2015, 0.002, 0.8, 20, 20)

for year in cYEARS:

    path_i = "./headlines/nytimes_headlines_histwords_" + str(year) + ".json"

    sentences = Headlines(path_i)

    model = gensim.models.Word2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS, alpha=INIT_LR,
                                   sentences=sentences, min_count=2, compute_loss=True, sg=False,
                                   callbacks=[callback(EPOCHS, INIT_LR)])

    models[year] = model

def cos_sim(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

wv_by_year = lambda word: lambda year: models[year].wv[word]

def compare_cos(w1, w2, year):
    wv1 = wv_by_year(w1)(year)
    wv2 = wv_by_year(w2)(year)

    return cos_sim(wv1, wv2)

# pre-alignment: loop through all years, check cosine similarity of sample words across each year
cWORDS = ['president', 'senator', 'obama', 'trump', 'politics', 'mask', 'flu', 'sick', 'shutdown']

com = 'president'
cos_sims_f = {}
for year in cYEARS:
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

between_years_df = pd.DataFrame(index=cYEARS, columns=cYEARS)
wv_com = wv_by_year(com)

for y1 in cYEARS:
    for y2 in cYEARS:
        between_years_df.at[y1, y2] = round(cos_sim(wv_com(y1), wv_com(y2)), 3)

print(between_years_df.to_string)




# ALIGNMENT
align_years(cYEARS) 


print("...POST ALIGNMENT...")
# then do same test post-alignment

wv_by_year2 = lambda word: lambda year: models[year + 20].wv[word]
def compare_cos2(w1, w2, year):
    wv1 = wv_by_year2(w1)(year)
    wv2 = wv_by_year2(w2)(year)

    return cos_sim(wv1, wv2)

com = 'president'
cos_sims_f = {}
for year in cYEARS:
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
between_years_df = pd.DataFrame(index=cYEARS, columns=cYEARS)
wv_com = wv_by_year2(com)

for y1 in cYEARS:
    for y2 in cYEARS:
        between_years_df.at[y1, y2] = round(cos_sim(wv_com(y1), wv_com(y2)), 3)

print(between_years_df.to_string)
