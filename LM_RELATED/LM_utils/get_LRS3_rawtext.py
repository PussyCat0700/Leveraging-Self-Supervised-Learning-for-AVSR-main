# 生成pretrain trainval.txt
from tqdm import tqdm

with open("/home/gryang/LRS3/pretrain.txt") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=line.strip()[2:]
        line= "/home/gryang"+line+".txt"
        with open(line) as f:
            lines1 = f.readline().strip()
            lines1=lines1.split(" ")[2:]
            word_list.extend(lines1)
  
with open("/home/gryang/LRS3/trainval.txt") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=line.strip()[2:]
        line= "/home/gryang"+line+".txt"
        with open(line) as f:
            lines1 = f.readline().strip()
            lines1=lines1.split(" ")[2:]
            word_list.extend(lines1)      
