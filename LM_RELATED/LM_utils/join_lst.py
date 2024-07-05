
from tqdm import tqdm

lexicon1="/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/librispeech_lexicon.lst"
lexicon2="/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/pretrain_trainval.lst"

dataset="plus"
object=dataset+".lst"

all_line=[]
with open(lexicon1) as f:
    lines = f.readlines()
    for line in tqdm(lines): 
        line=line.split("\t")
        # print(line)
        line=line[0]
        all_line.append(line)

with open(lexicon2) as f:
    lines = f.readlines()
    for line in tqdm(lines): 
        line=line.split("\t")
        # print(line)
        line=line[0]
        all_line.append(line)

word_set=list(set(all_line))      #1161
word_set=sorted(word_set)

with open(object, 'w') as f1:
    for word in tqdm(word_set):
        daxie= word.upper()
        l=list(daxie)
        char=" ".join(l)
        final= daxie + '\t'+ char +' |\n'
        f1.write(final)
