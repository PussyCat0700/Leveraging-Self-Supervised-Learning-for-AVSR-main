from itertools import groupby


import numpy as np
import torch
"""

from .flashlight_decoder import KenLMDecoder
from .viterbi_decoder import ViterbiDecoder
from .decoder_config import FlashlightDecoderConfig
from .flashlight_decoder import FairseqLMDecoder
"""
np.seterr(divide="ignore")

#debug    
#fairseq_dictionary= Dictionary.load('fairseq_dict.ltr.txt') #decoder= KenLMDecoder(cfg, fairseq_dictionary)  #decoder=ViterbiDecoder(dictionary) # npy_path= "/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/test.npy" # outputBatch= np.load(npy_path) # outputBatch= torch.from_numpy(outputBatch)   #(2, 1624, 32) # print(outputBatch.shape)
"""
def flash_infer(outputBatch, inputLenBatch, eosIx,cfg,logger, blank=0):   #batch_size=1的版本
    outputBatch = outputBatch.cpu() #(155,48,40) 
    inputLenBatch = inputLenBatch.cpu()  #(48)
    outputBatch[:, :, blank] = torch.log(torch.exp(outputBatch[:, :, blank]) + torch.exp(outputBatch[:, :, eosIx]))  
    reqIxs = np.arange(outputBatch.shape[2])  #array([0,...,39])
    reqIxs = reqIxs[reqIxs != eosIx]   #array([0,...,38])
    outputBatch = outputBatch[:, :, reqIxs]   #(152,48,39)    #得加上这些
      
    dictionary=Dictionary.load('/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/dict.ltr.txt')

    if cfg.type == "fairseqlm":
        cfg=FlashlightDecoderConfig.from_namespace(cfg)
        decoder= FairseqLMDecoder(cfg, dictionary)   
        outputBatch=outputBatch.transpose(0,1) #(10,155,39)   40
        result=decoder.decode(outputBatch)   
            
    else:  #kenlm 
        cfg=FlashlightDecoderConfig.from_namespace(cfg)
        #cfg = FlashlightDecoderConfig.create(cfg1)
        decoder= KenLMDecoder(cfg, dictionary)   #(155,10,39)  需要B T N 
        outputBatch=outputBatch.transpose(0,1) #(10,155,39)   40
        result=decoder.decode(outputBatch)   

    preds = list()
    predLens = list()
    
    for hypo in result:        
        hypo=hypo[0]
        
        hyp_words = " ".join(hypo["words"])
        print(hyp_words)
        
        pred=hypo["tokens"]

        pred=pred.int().cpu().tolist()

        # assert(pred[0]==1 and pred[-1]==1)    #确实全是这样所以        
        pred=pred[1:-1]

        pred.append(eosIx)  #最后加了一个39   len:36 type是np.aray([])  最后解码尾巴有很多1 这块可能得删？？？
        logger.info(pred)
        #print(pred)
        preds.extend(pred)
        predLens.append(len(pred)) 
            
    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch,hyp_words   #tensor([49, 49, 49, 46, 46, 46, 47, 47, 47, 48], dtype=torch.int32)
                
"""
def ctc_greedy_decode(outputBatch, inputLenBatch, eosIx, blank=0):
    """       # outputbatch(155,48,40)  #<EOS>index=39 blank=0
    Greedy search technique for CTC decoding.
    This decoding method selects the most probable character at each time step. This is followed by the usual CTC decoding
    to get the predicted transcription.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

    outputBatch = outputBatch.cpu() #(155,48,40) 
    inputLenBatch = inputLenBatch.cpu()  #(48) tensor([ 80,  80,  80,  80,  80,  80, 100, 151, 151, 155])
    outputBatch[:, :, blank] = torch.log(torch.exp(outputBatch[:, :, blank]) + torch.exp(outputBatch[:, :, eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])  #array([0,...,39])
    reqIxs = reqIxs[reqIxs != eosIx]   #array([0,...,38])
    outputBatch = outputBatch[:, :, reqIxs]   #(152,48,39)

    predCharIxs = torch.argmax(outputBatch, dim=2).T.numpy()  #(48,155)
    inpLens = inputLenBatch.numpy() #tensor([ 80,  80,  80,  80,  80,  80, 100, 151, 151, 155,  80,  80,  80, 155,84, 143, 155,  80,  80, 103, 104,  80,  80, 155,  80,  80,  95,  80,
    preds = list()
    predLens = list()
    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        ilen = inpLens[i]
        pred = pred[:ilen]  #取真实长度那么多 80
        pred = np.array([x[0] for x in groupby(pred)])  #相邻相同合并!
        pred = pred[pred != blank]  #去掉分隔符
        pred = list(pred) #len:35
        pred.append(eosIx)  #最后加了一个39   len:36 type是np.array([])
        preds.extend(pred)
        predLens.append(len(pred))   #最终preds就是一个list 长度603  predLens [36, 34, 44, 41, 22, 44, 77, 99, 112, 94]
    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch

"""
def flash_infer_store(outputBatch, inputLenBatch, eosIx, blank=0):   #batch_size=1的版本
    outputBatch = outputBatch.cpu() #(155,48,40) 
    inputLenBatch = inputLenBatch.cpu()  #(48)
    outputBatch[:, :, blank] = torch.log(torch.exp(outputBatch[:, :, blank]) + torch.exp(outputBatch[:, :, eosIx]))  
    reqIxs = np.arange(outputBatch.shape[2])  #array([0,...,39])
    reqIxs = reqIxs[reqIxs != eosIx]   #array([0,...,38])
    outputBatch = outputBatch[:, :, reqIxs]   #(152,48,39)    #得加上这些
      
    dictionary=Dictionary.load('dict.ltr.txt')
    
    cfg1={'_name': None, 'nbest': 1, 'unitlm': False, 'lmpath': '/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/4-gram.bin', 'lexicon': '/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/dict.ltr.txt', 'beam': 1500, 'beamthreshold': 100.0, 'beamsizetoken': None, 'wordscore': -1.0, 'unkweight': float('-inf'), 'silweight': 0.0, 'lmweight': 2.0, 'type': 'kenlm', 'unique_wer_file': True, 'results_path': None}
    cfg=FlashlightDecoderConfig(cfg1)  #nb 直接猜出用法哈哈哈 其实是不对的 直接改了default
    
    decoder= KenLMDecoder(cfg, dictionary)   #(155,10,39)  需要B T N 
    #decoder=ViterbiDecoder(dictionary)
    # print("viterbi")
    outputBatch=outputBatch.transpose(0,1) #(10,155,39)   40
    result=decoder.decode(outputBatch)
    
    for hypo in result:
        hypo=hypo[0]
        tokens=hypo["tokens"]
        print(tokens)
        print(len(hypo["tokens"]))
        hyp_pieces = dictionary.string(hypo["tokens"].int().cpu())  
        hyp_words = " ".join(hypo["words"])      
        print(hyp_words)
    preds = list()
    predLens = list()
    
    B=len(result)
    for i in range(B):
        hypo_dict= result[i][0]
        tokens=hypo_dict['tokens']
        #print(hypo_dict['words'])
        pred=tokens.numpy()
        pred=list(tokens)
        
        pred.append(eosIx)  #最后加了一个39   len:36 type是np.array([])  最后解码尾巴有很多1 这块可能得删？？？
        preds.extend(pred)
        predLens.append(len(pred)) 
    
    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()


    return predictionBatch, predictionLenBatch   #tensor([49, 49, 49, 46, 46, 46, 47, 47, 47, 48], dtype=torch.int32)
"""

def teacher_forcing_attention_decode(outputBatch, eosIx):
    outputBatch = outputBatch.cpu()
    predCharIxs = torch.argmax(outputBatch, dim=-1)
    seqLength = outputBatch.shape[1] - 1
    predictionBatch = []
    predictionLenBatch = []
    for pred in predCharIxs:
        firstEOSIx = seqLength if len((pred == eosIx).nonzero()) == 0 else (pred == eosIx).nonzero()[0]
        predictionBatch.append(pred[:firstEOSIx + 1] if pred[firstEOSIx] == eosIx else torch.cat((pred[:firstEOSIx + 1], torch.tensor([eosIx])), -1))
        predictionLenBatch.append(firstEOSIx + 1 if pred[firstEOSIx] == eosIx else firstEOSIx + 2)

    predictionBatch = torch.cat(predictionBatch, 0).int()
    predictionLenBatch = torch.tensor(predictionLenBatch)
    return predictionBatch, predictionLenBatch

# alpha 是 attentionOutLogProbs
def compute_CTC_prob(h, alpha, CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, numClasses, blank, eosIx):
    batch = h.shape[0]  #1  hbe本来是[39,1][39,2]...
    g = h[:, :, :, :-1]  # (1,1,39,1) 取了到-1（不含） 所以全是39  拆成g和c
    c = h[:, :, :, -1]  #(1,1,39)  [1,...,39] 取最后一项 所以是1到39
    alphaCTC = torch.zeros_like(alpha) #(1,1,39) 全0
    eosIxMask = c == eosIx #(1,1,39) 前面全是false 最后是true
    eosIxIndex = eosIxMask.nonzero()  #tensor([[ 0,  0, 38]])
    eosIxIndex = torch.cat((eosIxIndex[:, :1], torch.repeat_interleave((T - 1).unsqueeze(-1), numBeam, dim=0), eosIxIndex[:, 1:]), dim=-1).long()  #tensor([[ 0, 79,  0, 38]])
    eosIxIndex[:, -1] = 0  #tensor([[ 0, 79,  0,  0]])
    gamma_eosIxMask = torch.zeros_like(gamma_n).bool()  #（1，80，5，40）
    gamma_eosIxMask.index_put_(tuple(map(torch.stack, zip(*eosIxIndex))), torch.tensor(True)) #(1,80,5,40)
    alphaCTC[eosIxMask] = np.logaddexp(gamma_n[gamma_eosIxMask], gamma_b[gamma_eosIxMask])     #这块就是 if c=[EOS] log pctc(h|x)=log{ + }

    if g.shape[-1] == 1:
        gamma_n[:, 1, 0, 1:-1] = CTCOutLogProbs[:, 1, 1:-1]
    else:
        gamma_n[:, 1, :numBeam, 1:-1] = -np.inf
    gamma_b[:, 1, :numBeam, 1:-1] = -np.inf

    psi = gamma_n[:, 1, :numBeam, 1:-1]
    for t in range(2, T.max()):
        activeBatch = t < T
        gEndWithc = (g[:, :, :, -1] == c)[:, :, :-1].nonzero()
        added_gamma_n = torch.repeat_interleave(gamma_n[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1)
        if len(gEndWithc):
            added_gamma_n.index_put_(tuple(map(torch.stack, zip(*gEndWithc))), torch.tensor(-np.inf).float())
        phi = np.logaddexp(torch.repeat_interleave(gamma_b[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1), added_gamma_n)
        expandShape = [batch, numBeam, numClasses - 1]
        gamma_n[:, t, :numBeam, 1:-1][activeBatch] = np.logaddexp(gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch], phi[activeBatch]) \
                                                     + CTCOutLogProbs[:, t, None, 1:-1].expand(expandShape)[activeBatch]
        gamma_b[:, t, :numBeam, 1:-1][activeBatch] = \
            np.logaddexp(gamma_b[:, t - 1, :numBeam, 1:-1][activeBatch], gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch]) \
            + CTCOutLogProbs[:, t, None, None, blank].expand(expandShape)[activeBatch]
        psi[activeBatch] = np.logaddexp(psi[activeBatch], phi[activeBatch] + CTCOutLogProbs[:, t, None, 1:-1].expand(phi.shape)[activeBatch])

    return torch.cat((psi, alphaCTC[:, :, -1:]), dim=-1)

# alpha 是 attentionOutLogProbs
def my_compute_CTC_prob(h, alpha, CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, numClasses, blank, eosIx):
    batch = h.shape[0]  #1
    g = h[:, :, :, :-1]  # (1,1,39,1) 全是39
    c = h[:, :, :, -1]  #(1,1,39)  [1,...,39]
    alphaCTC = torch.zeros_like(alpha) #(1,1,39) 全0
    eosIxMask = c == eosIx #(1,1,39) 前面全是false 最后是true
    eosIxIndex = eosIxMask.nonzero()  #tensor([[ 0,  0, 38]])
    eosIxIndex = torch.cat((eosIxIndex[:, :1], torch.repeat_interleave((T - 1).unsqueeze(-1), numBeam, dim=0), eosIxIndex[:, 1:]), dim=-1).long()  #tensor([[ 0, 79,  0, 38]])
    eosIxIndex[:, -1] = 0  #tensor([[ 0, 79,  0,  0]])
    gamma_eosIxMask = torch.zeros_like(gamma_n).bool()  #（1，80，5，40）
    gamma_eosIxMask.index_put_(tuple(map(torch.stack, zip(*eosIxIndex))), torch.tensor(True)) #(1,80,5,40)
    alphaCTC[eosIxMask] = np.logaddexp(gamma_n[gamma_eosIxMask], gamma_b[gamma_eosIxMask])     #这块就是 if c=[EOS] log pctc(h|x)=log{ + }

    if g.shape[-1] == 1:
        gamma_n[:, 1, 0, 1:-1] = CTCOutLogProbs[:, 1, 1:-1]
    else:
        gamma_n[:, 1, :numBeam, 1:-1] = -np.inf
    gamma_b[:, 1, :numBeam, 1:-1] = -np.inf

    psi = gamma_n[:, 1, :numBeam, 1:-1]
    for t in range(2, T.max()):
        activeBatch = t < T
        gEndWithc = (g[:, :, :, -1] == c)[:, :, :-1].nonzero()
        added_gamma_n = torch.repeat_interleave(gamma_n[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1)
        if len(gEndWithc):
            added_gamma_n.index_put_(tuple(map(torch.stack, zip(*gEndWithc))), torch.tensor(-np.inf).float())
        phi = np.logaddexp(torch.repeat_interleave(gamma_b[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1), added_gamma_n)
        expandShape = [batch, numBeam, numClasses - 1]
        gamma_n[:, t, :numBeam, 1:-1][activeBatch] = np.logaddexp(gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch], phi[activeBatch]) \
                                                     + CTCOutLogProbs[:, t, None, 1:-1].expand(expandShape)[activeBatch]
        gamma_b[:, t, :numBeam, 1:-1][activeBatch] = \
            np.logaddexp(gamma_b[:, t - 1, :numBeam, 1:-1][activeBatch], gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch]) \
            + CTCOutLogProbs[:, t, None, None, blank].expand(expandShape)[activeBatch]
        psi[activeBatch] = np.logaddexp(psi[activeBatch], phi[activeBatch] + CTCOutLogProbs[:, t, None, 1:-1].expand(phi.shape)[activeBatch])
    return torch.cat((psi, alphaCTC[:, :, -1:]), dim=-1)