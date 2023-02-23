# compare batchsize_1 and batch_size 48

bs1="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/pred_HYBRID_1.txt"
bs48="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/pred_HYBRID_48.txt"
with open(bs1, 'r') as f:
    lines1 = f.readlines()

with open(bs48, 'r') as f1:
    lines48 = f1.readlines()
    
notequal=0
for line1,line48 in zip(lines1,lines48):
    if line1!=line48:
        notequal+=1
        print(line1)
        print(line48)
        
print(notequal)  #48
