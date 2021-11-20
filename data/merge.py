import os
from tqdm import tqdm

with open("../training") as f1,\
    open("../split/encoding") as f2,\
    open("encoding", "w") as fo:
        lines_f1 = f1.readlines()
        lines_f2 = f2.readlines()
        lines = list(lines_f2) + list(lines_f1)
        for line in tqdm(lines):
            fo.write(line.strip() + "\n")
