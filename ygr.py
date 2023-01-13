from flashlight.lib.text.dictionary import create_word_dict, load_words

lexicon='/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/librispeech_lexicon.lst'
lexicon = load_words(lexicon)

word_dict = create_word_dict(lexicon)

word_list=[]

with open("trgt.txt") as f:
    count=0
    lines = f.readlines()
    for line in lines:
        line=line.strip()
        line=line.split(" ")
        word_list.extend(line)
        
unk_word = word_dict.get_index("<unk>")  #28298

unk_count=0
word_idx = word_dict.get_index('CHABA')
word_idx = word_dict.get_index('AONE')
print(word_idx)