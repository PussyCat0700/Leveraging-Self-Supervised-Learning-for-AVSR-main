# 算OOC 生成lst 用val.txt 做模拟
from tqdm import tqdm
#from flashlight.lib.text.dictionary import create_word_dict, load_words

# lexicon='/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/librispeech_lexicon.lst'
# lexicon = load_words(lexicon)

# word_dict = create_word_dict(lexicon)

dataset="pretrain_trainval"
object=dataset+".lst"
print(object)
word_list=[]
with open("/data2/alumni/gryang/LRS3/pretrain.txt") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=line.strip()[2:]
        line= "/data2/alumni/gryang"+line+".txt"
        with open(line) as f:
            lines1 = f.readline().strip()
            lines1=lines1.split(" ")[2:]
            word_list.extend(lines1)
  
with open("/data2/alumni/gryang/LRS3/trainval.txt") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=line.strip()[2:]
        line= "/data2/alumni/gryang"+line+".txt"
        with open(line) as f:
            lines1 = f.readline().strip()
            lines1=lines1.split(" ")[2:]
            word_list.extend(lines1)      

print(len(word_list))  #val:3587
word_set=list(set(word_list))      #1161
print(len(word_set))  #(9890 但是又重复)
word_set=sorted(word_set)
with open(object, 'a') as f1:
    for word in tqdm(word_set):
        daxie= word.upper()
        l=list(daxie)
        char=" ".join(l)
        final= daxie + '\t'+ char +' |\n'
        f1.write(final)


"""
pretrain_trainval.lst
100%|██████████████████████████████████████████████████████████████████████████████████████| 118516/118516 [15:28<00:00, 127.67it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████| 31982/31982 [03:57<00:00, 134.41it/s]
4251740
51277
100%|█████████████████████████████████████████████████████████████████████████████████████| 51277/51277 [00:00<00:00, 877809.90it/s]
"""