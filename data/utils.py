import cv2 as cv
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from config import args
import time

def prepare_main_input(index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR):
    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """
    #print(targetFile)
    #print(index)

    with open(targetFile, "r") as f:
        trgt = f.readline().strip()[7:]  #'SO WE NEED YOU TO HELP US IN OUR REVIVAL CAMPAIGN'

        coun = trgt.count("{")
        for i in range(coun):
            left = trgt.find("{")
            if left != -1:
                right = trgt.find("}")
                trgt  = trgt .replace(trgt [left:right + 2], "")


    #print(trgt)
    trgtin = [charToIx[item] for item in trgt] #[8, 4, 1, 15, 2, 1, 7, 2, 2, 12, 1, 14, 4, 13, 1, 3, 4, 1, 9, 2, 11,
    trgtin.insert(0, charToIx["<EOS>"])  #[39,8,4,...]
    trgtout = [charToIx[item] for item in trgt]
    trgtout.append(charToIx["<EOS>"])   #[..,39] 在最后面加39
    trgtin = np.array(trgtin)
    trgtout = np.array(trgtout)
    trgtLen = len(trgtout)  #50

    # audio file
    if not modal == "VO":
        audInp = np.array(h5["flac"][index])  # ndarray(22528,)
        audInp = (audInp - audInp.mean()) / audInp.std()
        if noise is not None:
            pos = np.random.randint(0, len(noise) - len(audInp) + 1)
            noise = noise[pos:pos + len(audInp)]
            noise = noise / np.max(np.abs(noise))
            gain = 10 ** (noiseSNR / 10)
            noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
            audInp = audInp + noise
        audInp = torch.from_numpy(audInp)
    else:
        audInp = None

    # visual file
    if not modal == "AO":
        vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)  #(120,2040,3)
        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]  #(17,120,120)
        vidInp = torch.tensor(vidInp).unsqueeze(1)  #(17,1,120,120)
        vidInp = transform(vidInp) #(17,1,112,112)
    else:
        vidInp = None

    inp = (audInp, vidInp)
    trgtin = torch.from_numpy(trgtin)
    trgtout = torch.from_numpy(trgtout)
    trgtLen = torch.tensor(trgtLen)
    #print("main","trgtLen",trgtLen,"inp shape",inp[0].shape)

    return inp, trgtin, trgtout, trgtLen


def prepare_pretrain_input(index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR, numWordsRange, maxLength):  #(3,21)  160
    """
    Function to convert the data sample in the pretrain dataset into appropriate tensors.
    """

    # reading the whole target file and the target
    #print(targetFile)
    #print(index)
    # targetFile="/home/gryang/LRS3/pretrain/HdHW77blulg/00006.txt"
    # index=99639

    #print("start debug!")
                      #======================================
    try:
        with open(targetFile, "r") as f:
            lines = f.readlines()
    except:
        print("error")
        print(targetFile)
        print(index)
        return 0, 0, 0, 0

    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]

    coun = trgt.count("{")
    for i in range(coun):
        left = trgt.find("{")
        if left != -1:
            right = trgt.find("}")
            trgt = trgt.replace(trgt[left:right + 2], "")

    trgt=trgt.strip()
    # print(trgt)
    words = trgt.split(" ")
    #print("words",words)


    numWords = len(words) // 3
    if numWords < numWordsRange[0]:   #3   #（numwordsRange 是个tuple（3，21）
        numWords = numWordsRange[0]
    elif numWords > numWordsRange[1]:  #21
        numWords = numWordsRange[1]




    #count=0

    while True:
        #count+=1

        # if number of words in target is less than the required number of words, consider the whole target
        if len(words) <= numWords:


            trgtNWord = trgt

            # audio file
            if not modal == "VO":
                audInp = np.array(h5["flac"][index])
                audInp = (audInp - audInp.mean()) / audInp.std()
                if noise is not None:
                    pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                    noise = noise[pos:pos + len(audInp)]
                    noise = noise / np.max(np.abs(noise))
                    gain = 10 ** (noiseSNR / 10)
                    noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                    audInp = audInp + noise
                audInp = torch.from_numpy(audInp)
            else:
                audInp = None

            # visual file
            #t1 = time.time()
            if not modal == "AO":
                try:

                    vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                    vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
                    vidInp = torch.tensor(vidInp).unsqueeze(1)
                    vidInp = transform(vidInp)



                except:
                    print("error")
                    print(targetFile)
                    print(index)
                    return 0,0,0,0
            else:
                vidInp = None

            #t2=time.time()
            # print("<=t1",t2-t1)



        else:

            #t3 = time.time()
            # make a list of all possible sub-sequences with required number of words in the target
            nWords = [" ".join(words[i:i + numWords])
                      for i in range(len(words) - numWords + 1)]
            nWordLens = np.array(
                [len(nWord) + 1 for nWord in nWords]).astype(np.float)

            # choose the sub-sequence for target according to a softmax distribution of the lengths
            # this way longer sub-sequences (which are more diverse) are selected more often while
            # the shorter sub-sequences (which appear more frequently) are not entirely missed out
            ix = np.random.choice(np.arange(len(nWordLens)), p=nWordLens / nWordLens.sum())
            #print("ix",ix)
            #print("numWords",numWords)
            trgtNWord = nWords[ix]

            # reading the start and end times in the video corresponding to the selected sub-sequence
            startTime = float(lines[4 + ix].split(" ")[1])
            #print(startTime)
            endTime = float(lines[4 + ix + numWords - 1].split(" ")[2])
            # audio file


            #t4 = time.time()
            # print(">=t2:", t4 - t3)

            if not modal == "VO":
                samplerate = 16000
                audInp = np.array(h5["flac"][index])  #（81920，）
                audInp = (audInp - audInp.mean()) / audInp.std()
                if noise is not None:
                    pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                    noise = noise[pos:pos + len(audInp)]
                    noise = noise / np.max(np.abs(noise))
                    gain = 10 ** (noiseSNR / 10)
                    noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                    audInp = audInp + noise
                audInp = torch.from_numpy(audInp)
                audInp = audInp[int(samplerate * startTime):int(samplerate * endTime)]  #！！！！！！！
            else:
                audInp = None

            # visual file
            if not modal == "AO":
                videoFPS = 25
                try:
                    #t1 = time.time()
                    vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                    vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]  ##这一句报错x
                    vidInp = torch.tensor(vidInp).unsqueeze(1)
                    vidInp = transform(vidInp)
                    vidInp = vidInp[int(np.floor(videoFPS * startTime)): int(np.ceil(videoFPS * endTime))]
                   # t2=time.time()
                   # print(t2-t1)
                except:
                    print("error")
                    print(targetFile)
                    print(index)
                    return 0, 0, 0, 0

            else:
                vidInp = None

            # t5 = time.time()
            # print(">=t3:", t5 - t4)

        # converting each character in target to its corresponding index
        #print(trgtNWord)
        #trgtNWord=  trgtNWord.replace("{NS} ","")

        # t4 = time.time()
        # print("t2:", t4 - t3)


        trgtin = [charToIx[item] for item in trgtNWord]
        trgtout = [charToIx[item] for item in trgtNWord]
        trgtin.insert(0, charToIx["<EOS>"])
        trgtout.append(charToIx["<EOS>"])
        trgtin = np.array(trgtin)
        trgtout = np.array(trgtout)
        trgtLen = len(trgtout)

        inp = (audInp, vidInp)
        trgtin = torch.from_numpy(trgtin)
        trgtout = torch.from_numpy(trgtout)
        trgtLen = torch.tensor(trgtLen)
        # try:
        inpLen = len(vidInp) if not args["MODAL"] == "AO" else len(audInp) / 640
        # except:
        #     print("error")
        #     print(targetFile)
        #     print(index)

        if inpLen <= maxLength:   #maxlength:160
            break
        elif inpLen > maxLength + 80:
            numWords -= 2
        else:
            numWords -= 1


    # t2=time.time()
    # if t2-t1>=3: # 感觉这三者没有必然联系
    #     print("time:",t2-t1)
    # print("len(words)",len(words))
    # print("count:",count)


        #print("pretrain", "trgtLen", trgtLen, "inp shape", inp[0].shape)
    #print("==")
    return inp, trgtin, trgtout, trgtLen


def collate_fn(dataBatch):   #{list:4} 每一个是一个{tuple:4}  (none,62,1,112,112) (43,)(43,) 43 就是返回的那4样 测的时候设置的batch_size是4
           #也就是说先get_item 凑够数量 以4个为单位输进来
    """
    Collate function definition used in Dataloaders.
    """
    # audio & mask
    if not args["MODAL"] == "VO":
        aud_seq_list = [data[0][0] for data in dataBatch]
        aud_padding_mask = torch.zeros((len(aud_seq_list), len(max(aud_seq_list, key=len))), dtype=torch.bool)
        for i, seq in enumerate(aud_seq_list):
            aud_padding_mask[i, len(seq):] = True
        aud_seq_list = pad_sequence(aud_seq_list, batch_first=True)
    else:
        aud_seq_list = None
        aud_padding_mask = None
    # visual & len
    if not args["MODAL"] == "AO":
        vis_seq_list = pad_sequence([data[0][1] for data in dataBatch], batch_first=True)  #(4,147,1,112,112)   #pad_sequence((none,62,1,112,112))
        vis_len = torch.tensor([len(data[0][1]) for data in dataBatch]) #tensor([ 62,  62,  97, 147])
    else:
        vis_seq_list = None
        vis_len = None

    inputBatch = (aud_seq_list, aud_padding_mask, vis_seq_list, vis_len)

    targetinBatch = pad_sequence([data[1] for data in dataBatch], batch_first=True)
    targetoutBatch = pad_sequence([data[2] for data in dataBatch], batch_first=True)
    targetLenBatch = torch.stack([data[3] for data in dataBatch])

    return inputBatch, targetinBatch, targetoutBatch, targetLenBatch   #这里是真的batch那一步
