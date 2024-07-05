# 生成LRS2 train,val,test.txt
from tqdm import tqdm

#目录
pretraindir="/data2/alumni/xcpan/server_1/LRS2/mvlrs_v1/pretrain.txt"  #96318 行  #finish
traindir="/data2/alumni/xcpan/server_1/LRS2/mvlrs_v1/train.txt"  #45839 行  #finish
valdir="/data2/alumni/xcpan/server_1/LRS2/mvlrs_v1/val.txt"  #1082 行  #finish
testdir="/data2/alumni/xcpan/server_1/LRS2/mvlrs_v1/test.txt"   #1243行    #finish

# 96318+45839=142157
maindir="/data2/alumni/xcpan/server_1/LRS2/mvlrs_v1/pretrain/"   #真的数据在哪
# val和test的数据与pretrain 和train独立 且序号大于他们

# pretrain 553-629
# train  553-629
# val 630-632
# test 633-639

#main 文件夹 553-639  check过val在 test也在

#写到哪
valobj="/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS2/val.txt" #pretrain+train
testobj="/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS2/test.txt"
trainobj="/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS2/train.txt" #pretrain+train
pretrainobj="/data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS2/pretrain.txt" #pretrain+train

object=pretrainobj
print("write:",object)
with open(object, 'w') as final:
    with open(pretraindir) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line=line.strip()  #['6330311066473698535/00011', 'NF\n']
            line=maindir+line+'.txt'  #/data2/alumni/xcpan/server_1/LRS2/mvlrs_v1/main/6300370419826092098/00001.txt
            #print(line)

            with open(line) as f:
                lines1 = f.readline().strip()
                lines1=lines1[7:]+'\n'
                #print(lines1)
                final.write(lines1)


# test  test和别的格式不一样
# with open(object, 'w') as final:
#     with open(testdir) as f:
#         lines = f.readlines()
#         for line in tqdm(lines):
#             line=line.split(" ")  #['6330311066473698535/00011', 'NF\n']
#             line=maindir+line[0]+'.txt'  #/data2/alumni/xcpan/server_1/LRS2/mvlrs_v1/main/6330311066473698535/00011
#             with open(line) as f:
#                 lines1 = f.readline().strip()
#                 lines1=lines1[7:]+'\n'
#                 final.write(lines1)


  