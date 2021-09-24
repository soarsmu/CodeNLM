import reader
import codeprep.api.text as cp

from nltk.corpus import wordnet as wn

word_to_id, id_to_word = reader._read_vocab("./vocab.txt")

words = [n for n in wn.all_lemma_names() if len(n.split("_")) < 2]

f_t = 0
for index, w in enumerate(words):
    unk = 0
    t = 0
    w_split = cp.bpe(w, bpe_codes_id='10k')
    for w in w_split:
        # if w == "<w>" or w == "</w>":
        #     continue
        # elif w in word_to_id.keys():
        t += 1
        # else: unk += 1

    if t - unk > 0:
        # print(1.0/(t - unk))
        f_t += 1.0/(t - unk)
    else:
        continue
    if index % 1000 == 0:
        print(index)
        print(f_t)

print(f_t)

        




