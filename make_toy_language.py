### Make toy language with markov property
import math
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

# TBD: add Gaussian noise?

### Generate k years of 'text'
def generate_k_texts(k = 50, text_length = 10000, combo=False, combo_word = 'F', combo_freq = 0.4):

    texts = []

    combo_words = []
    if combo:
        combo_words = random.sample(words, k = 2)
        print("Adding combination of", combo_words, "as", combo_word)

        # set shift params
        midpoint = random.choice(range(k))
        s = random.uniform(0, k/2) / k

        print("shift parameters: m =", midpoint, "& s = ", s)

    for year in range(k):

        t = []
        cur = bos_word
        for _ in range(text_length):
            probs = pr[cur]

            next = random.choices(words, weights=probs)[0]

            t.append(next)
            cur = next

        # substitute combined word in if needed
        if combo:

            # shift = \sigma( s * (t - m) )
            shift = 1.0 / (1 + math.exp(-(s * (year - midpoint))))
            shift_fst = shift * combo_freq
            shift_scd = (1 - shift) * combo_freq

            for i in range(len(t)):

                w_i = t[i]
                sub_prob = random.random()

                should_subst_first = w_i == combo_words[0] and sub_prob < shift_fst
                should_subst_second = w_i == combo_words[1] and sub_prob < shift_scd

                if should_subst_first or should_subst_second:

                    t[i] = combo_word
            

        texts.append(" ".join(t))

    return texts

ts = generate_k_texts(combo=True)

# show probabilities
print("probabilities:")
for word in words:
    print(word, "::", pr[word])
