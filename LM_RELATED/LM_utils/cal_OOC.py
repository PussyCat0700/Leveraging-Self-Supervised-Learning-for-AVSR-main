# 算OOC
from tqdm import tqdm
from flashlight.lib.text.dictionary import create_word_dict, load_words

lexicon="/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/pretrain_trainval.lst"
#lexicon='/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/librispeech_lexicon.lst'
lexicon = load_words(lexicon)

word_dict = create_word_dict(lexicon)

word_list=[]
with open("/data2/alumni/gryang/LRS3/test.txt") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=line.strip()[2:]
        line= "/data2/alumni/gryang"+line+".txt"
        with open(line) as f:
            lines1 = f.readline().strip()
            lines1=lines1.split(" ")[2:]
            word_list.extend(lines1)
        
   
print(len(word_list))  #9890
word_set=list(set(word_list))      #2001
print(len(word_set)) 
#print(word_set)

unk_word = word_dict.get_index("<unk>")  
print("unk_word:",unk_word) 
unk_count=0

for word in word_set:
    word_idx = word_dict.get_index(word)
    if word_idx==unk_word:
        unk_count+=1
print("set: unk:%d oov:%f"%(unk_count,float(unk_count)/len(word_set)))

for word in word_list:
    word_idx = word_dict.get_index(word)
    if word_idx==unk_word:
        unk_count+=1
print("list: unk:%d oov:%f"%(unk_count,float(unk_count)/len(word_list)))  

# 9890
# 2001
# set: unk:56 oov:0.027986
# list: unk:122 oov:0.012336


#新的lst
# 9890
# 2001
# 25595
# set: unk:16 oov:0.007996
# list: unk:32 oov:0.003236



#LRS23.lst
# 9890
# 2001
# unk_word: 23270
# set: unk:9 oov:0.004498
# list: unk:18 oov:0.001820

# LRS23_librispeech.lst
# 9890
# 2001
# unk_word_id: 212614
# set: unk:4 oov:0.001999
# list: unk:8 oov:0.000809