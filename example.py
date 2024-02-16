from representations.sequentialembedding import SequentialEmbedding

"""
Example showing how to load a series of historical embeddings and compute similarities over time.
Warning that loading all the embeddings into main memory can take a lot of RAM
"""

if __name__ == "__main__":
    fiction_embeddings = SequentialEmbedding.load("embeddings/nytimes", range(2013, 2024, 1))
    words = ("president", "alaska")
    time_sims = fiction_embeddings.get_time_sims(words[0], words[1])  
    print "Similarity between", words[0], "and", words[1], "drastically increases from 2010s to the 2020s:"
    for year, sim in time_sims.iteritems():
        print "{year:d}, cosine similarity={sim:0.2f}".format(year=year,sim=sim)
