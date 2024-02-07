from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import logging
import tempfile
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        ARCHIVE_PATH = './nytimes_headlines_histwords_2013.json'

        with open(ARCHIVE_PATH) as file:
            articles = json.load(file)

            for ar in articles:
                headline = ar['headline'].encode('utf-8')
                yield utils.simple_preprocess(headline)


sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, min_count=10, workers=4)

with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    model.save(temporary_filepath)