from gensim import utils
import gensim.models
import logging
import json
import numpy as np
import gensim.downloader as api
from gensim.test.utils import datapath

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

sentences = MyCorpus()

sentences = gensim.models.word2vec.Text8Corpus('./synth_task/text8')
model = gensim.models.Word2Vec(sentences=sentences, vector_size=200)

print(model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))

print(model.wv.most_similar(['girl', 'father'], ['boy'], topn=3))

print(model.wv.doesnt_match("breakfast cereal dinner lunch".split()))

more_examples = ["he his she", "big bigger bad", "going went being"]

for example in more_examples:
    a, b, x = example.split()
    predicted = model.wv.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))  




model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

'''
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model)

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    print("matplotlibbing")

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

#try:
#    get_ipython()
#except Exception:
plot_function = plot_with_matplotlib
#else:
#    plot_function = plot_with_plotly

plot_function(x_vals, y_vals, labels)
'''