### Make toy language with markov property
import json
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

# TBD: add Gaussian noise?

### Generate k years of 'text'
def generate_k_texts(words=[], 
                     probs={}, 
                     bos="", 
                     k = 50, 
                     text_length = 10000, 
                     combo=False, 
                     combo_word = 'I', 
                     combo_freq = 0.4, 
                     trigram=False):

    texts = []

    combo_words = []
    if combo:
        combo_words = random.sample(words, k = 2)
        print("Adding combination of", combo_words, "as", combo_word)

        # set shift params
        midpoint = random.choice(range(k))
        s = random.random()

        print("shift parameters: m =", midpoint, "& s = ", s)

    for year in range(k):

        t = []
        cur = bos_word
        for _ in range(text_length):
            prob_next = probs[cur]

            next = random.choices(words, weights=prob_next)[0]

            t.append(next)
            if trigram:
                # NB: assumes one-char words
                cur = cur[-1] + next
            else:
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

    return (texts, combo_words, midpoint, s)

words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
bos_word = '-'

COMBO_WORD = 'I'
COMBO_FREQ = 0.5
YEARS = 20
TXT_LEN = 10000

USE_TRIGRAM = False

num_words = len(words)

# init transition probabilities for all words + bos
pr = {}
pr[bos_word] = get_probs(num_words)

for word in words:
    pr[word] = get_probs(num_words)
    
    # for trigram model, add probabilities for bigrams
    if USE_TRIGRAM:
        # bigrams from bos
        pr[bos_word + word] = get_probs(num_words)

        # other bigrams
        for word2 in words:
            pr[word + word2] = get_probs(num_words)

# show probabilities
print("probabilities:")
print(pr)

ts, combos, m, s = generate_k_texts(words=words, 
                      probs=pr, 
                      bos=bos_word, 
                      combo=True, 
                      k=YEARS,
                      text_length=TXT_LEN,
                      combo_word=COMBO_WORD, 
                      combo_freq=COMBO_FREQ,
                      trigram=USE_TRIGRAM)

### SAVE TEXTS + PARAMETERS
 
# parameters
params = {
    'words': words,
    'bos': bos_word,
    'probabilities': pr,
    'years': YEARS,
    'text_length': TXT_LEN,
    'combined_words': combos,
    'combo_word': COMBO_WORD,
    'combo_freq': COMBO_FREQ,
    'm': m,
    's': s,
}

json_file = open("./synth_task/toy_lang_params.json", "w") 
   
json.dump(params, json_file, indent = 4) 
   
json_file.close() 

# save texts
SENTENCE_LEN = 500
for year in range(YEARS):

    text_i = ts[year]

    text_json = [{'text': text_i[i : i + SENTENCE_LEN]} for i in range(int(TXT_LEN / SENTENCE_LEN))]

    text_json_file = open("./synth_task/toy_lang_" + str(year) + ".json", "w")

    json.dump(text_json, text_json_file, indent = 4) 
   
    text_json_file.close() 



    
