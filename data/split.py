import re
import json
import time
import string
import logging
import argparse

import multiprocessing as mp

manager = mp.Manager()
q_to_store = manager.Queue()

from tqdm import tqdm
from spiral import ronin

keywords = json.load(open("keywords.json"))

def identifier_split(line):
    split_data = []
    for tok in line.strip().split(" "):
        if tok in keywords["keywords"] or tok.isdigit() or tok in string.punctuation or tok[0] in string.punctuation:
            split_data.append(tok)
            continue
        split_words = []
        tok = re.split("([a-zA-Z0-9]+|\W+)", tok)

        for s in tok:
            if not s == "":
                if (s.isalnum() or s.isalpha()) and len(s)>2:
                    for _ in ronin.split(s): split_words.append(_)
                else:
                    split_words.append(s)

        if len(split_words) >= 2:
            for i, w in enumerate(split_words):
                if i < len(split_words) - 1:
                    split_data.append(w)
                    split_data.append("</w>")
                else:
                    split_data.append(w)
        elif len(split_words) > 0:
            split_data.append(split_words[0])
    # return " ".join(split_data)
    # # for i in split_data:
    # #     print(i)
    q_to_store.put(" ".join(split_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", "-i")
    parser.add_argument("--out_fp", "-o")

    args = parser.parse_args()

    with open(args.input_fp, "r") as f, \
            open(args.out_fp, "w") as fout:

        logging.info("Start to process files...")
        lines = f.readlines()
        pbar = tqdm(total=len(lines))
        update = lambda *args: pbar.update()

        start_time = time.time()
        pool = mp.Pool(mp.cpu_count())
        for line in lines:
            # all_tokens.append(identifier_split(line, args.split))
            pool.apply_async(identifier_split, args=(line,), callback=update)
        pool.close()
        pool.join()
        
        logging.info("Time cost: {} s".format(str(time.time()-start_time)))
        logging.info("Start to write files...")

        while not q_to_store.empty():
            single_data = q_to_store.get()
            if len(single_data):
                fout.write(single_data + "\n")
        logging.info("Done")


if __name__ == "__main__":
    main()

