### Make toy language with markov property
import random

# make transition probabilities
def get_probs(num_words=5):

    trs = []
    for _ in range(num_words - 1):
        trs.append(random.random())

    trs.sort()
    prs = [trs[0]]

    for i in range(1, num_words - 1):
        prs.append(trs[i] - trs[i - 1])

    prs.append(1.0 - trs[num_words - 2])
        
    return prs


words = ['a', 'b', 'c', 'd', 'e']
bos_word = '-'

pr = {}

# init transition probabilities for all words + bos
pr[bos_word] = get_probs()

for word in words:
    pr[word] = get_probs()

### Generate 'texts'
def generate_k_texts(k = 1000, text_length = 100):

    texts = []

    for _ in range(k):

        t = []
        cur = bos_word
        for _ in range(text_length):
            probs = pr[cur]

            next = random.choices(words, weights=probs)[0]

            t.append(next)
            cur = next

        texts.append(" ".join(t))

    return texts

ts = generate_k_texts(5, 20)

print(ts, pr['a'])