import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import prepare_pretrain_input, prepare_main_input
from .vision_transform import ToTensor, Normalize, RandomCrop, CenterCrop, RandomHorizontalFlip

import time
def get_files(datadir, dataset, fold):
    with open(datadir + "/" + dataset + ".txt", "r") as f:
        lines = f.readlines()
    datalist = [datadir + "/" + fold + "/" + line.strip().split(" ")[0] for line in lines]
    return datalist

def get_files1(datadir, dataset, fold):
    with open("/data2/alumni/gryang/"+datadir + "/" + dataset + ".txt", "r") as f:
        lines = f.readlines()
    #datalist = [datadir + "/" + fold + "/" + line.strip().split(" ")[0] for line in lines]
    datalist = ["/data2/alumni/gryang/"+ line.strip().split(" ")[0][3:] for line in lines]
    #datalist = [ line.strip().split(" ")[0][3:] for line in lines]
    # print(datalist[0])
    # exit(0)
    return datalist


class LRS2(Dataset):
    """
    A custom dataset class for the LRS2 dataset.
    """

    def __init__(self, modal, dataset, datadir, h5file, charToIx, stepSize, lrs2Aug, noiseParams):
        super(LRS2, self).__init__()
        self.modal = modal
                                    #第二个 ../LRS3/trainval/00j9bKdiOjk/50001'
        self.dataset = dataset    #第二个是从train.txt里读的，'/data2/alumni/xcpan/LRS2/mvlrs_v1/main/5535415699068794046/00001'
        if self.dataset == "train":  #../LRS3/pretrain/WQG8EJTY48M/00016  '/data2/alumni/xcpan/LRS2/mvlrs_v1/pretrain/5535415699068794046/00001'
            self.datalist = get_files1(datadir, 'pretrain', 'pretrain')  + get_files1(datadir, 'train', 'main')   #[list:142157: ['/data2/alumni/xcpan/LRS2/mvlrs_v1/pretrain/5535415699068794046/00001', '/data2/alumni/xcpan/LRS2/mvlrs_v1/pretrain/5535415699068794046/00002',
            #self.datalist = get_files1(datadir, 'pretrain', 'pretrain') + get_files1(datadir, 'train', 'main')
        elif self.dataset == "val":
            self.datalist = get_files1(datadir, 'val', 'main')  #'../LRS3/trainval/0GL5r3HVAZ0/50013'
        else:
            self.dataset = "test"
            self.datalist = get_files1(datadir, 'test', 'main')

        self.h5file = h5file   #'/data2/alumni/xcpan/LRS3.h5'
        with h5py.File(noiseParams["noiseFile"], "r") as f:  #{'noiseFile': '/data2/alumni/xcpan/LRS2/mvlrs_v1/Noise.h5', 'noiseProb': 0.25, 'noiseSNR': 5}
            self.noise = f["noise"][0]  #ndarray:57600000
        self.noiseSNR = noiseParams["noiseSNR"]
        self.noiseProb = noiseParams["noiseProb"]
        self.charToIx = charToIx
        self.stepSize = stepSize  #16384
        if lrs2Aug:
            self.transform = transforms.Compose([
                ToTensor(),
                RandomCrop(112),
                RandomHorizontalFlip(0.5),
                Normalize(mean=[0.4161], std=[0.1688])
            ])
        else:
            self.transform = transforms.Compose([
                ToTensor(),
                CenterCrop(112),
                Normalize(mean=[0.4161], std=[0.1688])
            ])
        return

    def open_h5(self):
        self.h5 = h5py.File(self.h5file, "r")

    def __getitem__(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()

        if self.dataset == "train":   #index=610
            # index goes from 0 to stepSize-1
            # dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            # fetch the sample at position 'index' in this randomly selected partition
            base = self.stepSize * np.arange(int(len(self.datalist) / self.stepSize) + 1)   # datalist, 118516 应该全是pretrain的 从pretrain.txt 搞出来的 # stepsize 16384
            ixs = base + index                        # [  0  16384  32768  49152  65536  81920  98304 114688 131072 147456]
            ixs = ixs[ixs < len(self.datalist)]     #[   610  16994  33378  49762  66146  82530  98914 115298]
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)  #以某种方式随机采样  #33378
        # passing the sample files and the target file paths to the prepare function to obtain the input tensors
        if index==99639 or index== 71740 or index==19753 or index==14116 or index==49729 or index==26726:
            index+=1

        targetFile = self.datalist[index] + ".txt"   #!!!!!!!!
        #print(targetFile)    #LRS3 pretrain 118516  train 31662  val 320  test 1321
        if self.dataset == "val":   #  LRS2 : pretrain 96318   train 45839    val 1082       test 1243
            index += 150178          #这俩数也得改  142157  =96318+45839 =pretrain+train
        elif self.dataset == "test":
            index += 150498      #原本  143239 = 96318+45839+1082=pretrain+train+val

        if np.random.choice([True, False], p=[self.noiseProb, 1 - self.noiseProb]):
            noise = self.noise
        else:
            noise = None

        #t1=time.time()

        if index < 118516:     #原本是96318   查过了 这个数确实是lrs2的那个行数 也就是文件数  原本应该是pretrain处理的 有一部分搞到main处理了 所以没有crop 导致超过500
            inp, trgtin, trgtout, trgtLen = prepare_pretrain_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)
            if inp==0 and trgtin ==0 and  trgtout ==0 and trgtLen==0:
                index+=1
                targetFile = self.datalist[index] + ".txt"
                inp, trgtin, trgtout, trgtLen = prepare_pretrain_input(index, self.modal, self.h5, targetFile,self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)

        else:
            inp, trgtin, trgtout, trgtLen = prepare_main_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR)

        #t2=time.time()
        # if t2-t1>=1:
        #     print("cost:",t2-t1)
        #     print(index,targetFile)
        return inp, trgtin, trgtout, trgtLen   #VO (none,(72,1,112,112) )

    def __len__(self):
        # each iteration covers only a random subset of all the training samples whose size is given by the step size   step size的作用在这里 感觉也没什么大用
        # this is done only for the pretrain set, while the whole val/test set is considered
        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)
