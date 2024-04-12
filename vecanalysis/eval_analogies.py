from gensim.models import KeyedVectors
import csv
# NB: should come from saved unaligned year-wise models

#### ANALOGY EVAL ####
def eval_analogy(models_path, an1, an2, top_knn=1):
    # top_knn chooses how many nearest neighbors to test against. 
    # Original model only tested closest neighbor but given noisiness of word vectors this seems misguided.

    w1, y1 = an1.split('-')
    w2, y2 = an2.split('-')

    # load models
    mod_y1 = KeyedVectors.load_word2vec_format(models_path + 'sgns_vectors_' + y1, binary=False)
    mod_y2 = KeyedVectors.load_word2vec_format(models_path + 'sgns_vectors_' + y2, binary=False)

    # align mod_y2 with y1
    mod_y2_aligned = smart_procrustes_align_gensim(mod_y1, mod_y2)

    # lookup word in y1
    wv1 = mod_y1.wv[w1].syn0norm

    # find nearest neighbor in y2
    preds = mod_y2_aligned.most_similar(wv1, top_knn)

    w2_preds = [wrd for wrd, _ in preds]
    
    return w2 in w2_preds

def eval_baseline(an1, an2):
    return an1.split('-')[0] == an2.split('-')[1]

#### READ IN TEMPORAL ANALOGIES ####
def load_analogies(a_path):
    analogies = []

    with open(a_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            analogies.append(row)

    return analogies

#### TEST ANALOGIES ###
def eval_analogies(analogies, models_path, top_knn=1):
    acc_emb = 0.0
    acc_base = 0.0
    for an1, an2 in analogies:

        if eval_analogy(models_path, an1, an2, top_knn=top_knn):
            acc_emb += 1

        if eval_baseline(an1, an2):
            acc_base += 1

    acc_base /= len(analogies)
    acc_emb /= len(analogies)
    print("overall accuracy:")
    print("embeddings:", acc_emb)
    print("baseline:", acc_base)

MODELS_PATH = './embeddings/nytimes-big/'
ANALOGIES_PATH = ''

TOP_KNN = 1

analogies = load_analogies(ANALOGIES_PATH)

eval_analogies(analogies, MODELS_PATH, TOP_KNN)