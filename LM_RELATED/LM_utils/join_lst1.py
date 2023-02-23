from tqdm import tqdm

lexicon1="/mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/librispeech_lexicon.lst"
lexicon2="/mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst"


dataset="23plus"
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

print(len(all_line))  #266005
word_set=list(set(all_line))    
word_set=sorted(word_set) 
print(len(word_set))  #219927

with open(object, 'w') as f1:
    for word in tqdm(word_set):
        daxie= word.upper()
        l=list(daxie)
        char=" ".join(l)
        final= daxie + '\t'+ char +' |\n'
        f1.write(final)
