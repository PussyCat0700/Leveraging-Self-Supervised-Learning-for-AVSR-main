# 算OOC
from tqdm import tqdm
from flashlight.lib.text.dictionary import create_word_dict, load_words

lexicon="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/pretrain_trainval.lst"
#lexicon='/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/librispeech_lexicon.lst'
lexicon = load_words(lexicon)

word_dict = create_word_dict(lexicon)

word_list=[]
with open("/home/gryang/LRS3/test.txt") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line=line.strip()[2:]
        line= "/home/gryang"+line+".txt"
        with open(line) as f:
            lines1 = f.readline().strip()
            lines1=lines1.split(" ")[2:]
            word_list.extend(lines1)
        
   
print(len(word_list))  #9890
word_set=list(set(word_list))      #2001
print(len(word_set))  #(9890 但是又重复)
#print(word_set)

unk_word = word_dict.get_index("<unk>")  #28298
print("unk_word:",unk_word) #25595
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