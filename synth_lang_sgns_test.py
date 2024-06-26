from gensim import utils
import gensim.models
import logging
import json
import numpy as np
import pandas as pd

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Sentences:
    """An iterator that yields sentences (lists of str)."""

    path = "toy_lang_0.json"

    def __init__(self, path):
        self.path = path

    def __iter__(self):

        with open(self.path) as file:
            articles = json.load(file)

            for ar in articles:
                yield ar['text'].split()



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
    base_embed = models[years[-1]]

    for year in years[:-1]:

        year_embed = models[year]

        print("Aligning year:", year)
        aligned_embed = smart_procrustes_align_gensim(base_embed, year_embed)

        print("Writing year:", year)
        models[year + years[-1] + 1] = aligned_embed
        aligned_embed.wv.save_word2vec_format('./embeddings/nytimes/sgns_vectors_' + str(year) + '.txt', binary=False)

    year = years[-1]
    print("Writing year:", year)
    models[year + years[-1] + 1] = base_embed
    

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
        #print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        
        # add final loss to grid search
        if (self.epoch, self.lr) not in losses_by_hyp:
            losses_by_hyp[(self.epoch, self.lr)] = 0.0
        
        losses_by_hyp[(self.epoch, self.lr)] += loss_now if loss_now > 0.0 else 100.0  
            
        self.epoch += 1

def grid_search_hyperparams(years, minLR, maxLR, num_lr_steps, eps):
    
    # search over all years to avoid overfitting on one year
    print("grid searching:")
    for year in years:
        print(year, "...")
        path_i = "./synth_task/toy_lang_" + str(year) + ".json"

        # train models for all ranges, combos of LR, Epochs
        lr_step = (maxLR - minLR) / num_lr_steps
        for lr in [minLR + i*lr_step for i in range(num_lr_steps)]:
            sentences = Sentences(path_i)

            model = gensim.models.Word2Vec(vector_size=VECTOR_SIZE, epochs=eps, alpha=lr,
                                        sentences=sentences, min_count=1, compute_loss=True, sg=False,
                                        callbacks=[callback(eps, lr)])
        
    # find hyperparams with best loss
    best_eps, best_lr = min(losses_by_hyp, key=losses_by_hyp.get)
    print("best hyperparams:", best_eps, "epochs &",  best_lr, "lr; with final average loss of", losses_by_hyp[(best_eps, best_lr)] / len(years))

    return best_eps, best_lr

# load parameters
params = None
with open('./synth_task/toy_lang_params.json') as file:
    params = json.load(file)

fst, snd = params['combined_words']
com = params['combo_word']
cYEARS = range(params['years'])
cWORDS = params['words'] + [com]
EPOCHS = 12
INIT_LR = 0.078
VECTOR_SIZE = 6

losses_by_hyp = {}

models = {}
    
# EPOCHS, INIT_LR = grid_search_hyperparams(cYEARS, 0.001, 0.5, 20, 15)

for year in cYEARS:

    path_i = "./synth_task/toy_lang_" + str(year) + ".json"

    sentences = Sentences(path_i)

    model = gensim.models.Word2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS, alpha=INIT_LR,
                                   sentences=sentences, min_count=1, compute_loss=True, sg=False,
                                   callbacks=[callback(EPOCHS, INIT_LR)])

    models[year] = model

from numpy.linalg import norm
def cos_sim(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

get_wv = lambda w: lambda year: models[year].wv[w]

def compare_cos(w1, w2, year):
    wv1 = get_wv(w1)(year)
    wv2 = get_wv(w2)(year)

    return cos_sim(wv1, wv2)

a_v = get_wv(fst)
combo = get_wv(com)
print("cosine sim (unaligned):", cos_sim(a_v(cYEARS[0]), a_v(cYEARS[-1])))

align_years(cYEARS) 

y_fst = cYEARS[0] + cYEARS[-1] + 1
y_lst = cYEARS[-1] + cYEARS[-1] + 1

print("cosine sim first combo word:", cos_sim(a_v(y_fst), a_v(y_lst)))
print("cosine sim combo with itself, years 1, 20:", cos_sim(combo(y_fst), combo(y_lst)))
print("...... now for combos ......")

cos_sims_f = {}
for year in [y_fst, y_lst]:
    cos_sims = {}
    for w2 in [fst, snd, com]:
        cos_sim_f = []
        for word in params['words']:
            cos_sim_f.append(round(compare_cos(word, w2, year), 3))
        
        if w2 == com:
            cos_sims_f[year] = cos_sim_f
        
        cos_sims[w2] = cos_sim_f
        
        print("for", year, w2, "has similarities of:", cos_sim_f)

    print("comparing the pairs:")
    for w in [fst, snd]:
        com_sim = cos_sims[com]
        w_sim = cos_sims[w]
        sim_diffs = [round(w_sim[i] - com_sim[i], 3) for i in range(len(w_sim))]
        print("for", year, w, "vs", com, ":", sim_diffs)
    print("......")
 
 
print("...POST ALIGNMENT...")
# then do same test post-alignment

wv_by_year2 = lambda word: lambda year: models[year + cYEARS[-1] + 1].wv[word]
def compare_cos2(w1, w2, year):
    wv1 = wv_by_year2(w1)(year)
    wv2 = wv_by_year2(w2)(year)

    return cos_sim(wv1, wv2)

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

    print("for pairs against combo:")
    for w in cWORDS:
        com_sim = cos_sims[com]
        w_sim = cos_sims[w]
        sim_diffs = [round(w_sim[i] - com_sim[i], 3) for i in range(len(w_sim))]
        print("for", year, w, "vs", com, ":", sim_diffs)
    print("......")

print("between years for combination char:")
between_years_df = pd.DataFrame(index=cYEARS, columns=[cYEARS[-1]])
wv_com = wv_by_year2(com)

for y1 in cYEARS:
    between_years_df.at[y1, cYEARS[-1]] = round(cos_sim(wv_com(y1), wv_com(cYEARS[-1])), 3)

print(between_years_df.to_string)