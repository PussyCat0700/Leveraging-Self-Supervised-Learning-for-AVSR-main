#import fairseq
from fairseq.checkpoint_utils import load_model_ensemble_and_task  
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.decoders import compute_CTC_prob
from .moco_visual_frontend import MoCoVisualFrontend
from .utils import PositionalEncoding, conv1dLayers, outputConv, MaskedLayerNorm, generate_square_subsequent_mask

from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from config import args
from operator import itemgetter
import math
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.models.lstm_lm import LSTMLanguageModel
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers import decoders

class AVNet(nn.Module):  

    def __init__(self, modal, W2Vfile, MoCofile, reqInpLen, modelargs):
        super(AVNet, self).__init__()
        dModel, nHeads, numLayers, peMaxLen, audinSize, vidinSize, fcHiddenSize, dropout, numClasses = modelargs
        self.modal = modal
        self.numClasses = numClasses
        self.reqInpLen = reqInpLen
        # A & V Modal
        tx_norm = nn.LayerNorm(dModel)
        self.maskedLayerNorm = MaskedLayerNorm()
        if self.modal == "AV":
            self.ModalityNormalization = nn.LayerNorm(dModel)
        self.EncoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        # audio
        if not self.modal == "VO":
            # front-end
            wav2vecModel, cfg, task = load_model_ensemble_and_task([W2Vfile], arg_overrides={
                "apply_mask": True,
                "mask_prob": 0.5,
                "mask_channel_prob": 0.25,
                "mask_channel_length": 64,
                "layerdrop": 0.1,
                "activation_dropout": 0.1,
                "feature_grad_mult": 0.0,
            })
            wav2vecModel = wav2vecModel[0]
            wav2vecModel.remove_pretraining_modules()
            self.wav2vecModel = wav2vecModel
            # back-end
            self.audioConv = conv1dLayers(self.maskedLayerNorm, audinSize, dModel, dModel, downsample=True)
            audioEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
            self.audioEncoder = nn.TransformerEncoder(audioEncoderLayer, num_layers=numLayers, norm=tx_norm)
        else:
            self.wav2vecModel = None
            self.audioConv = None
            self.audioEncoder = None
        # visual
        if not self.modal == "AO":
            # front-end
            visualModel = MoCoVisualFrontend()
            if MoCofile is not None:
                visualModel.load_state_dict(torch.load(MoCofile, map_location="cpu"), strict=False)
            self.visualModel = visualModel
            # back-end
            self.videoConv = conv1dLayers(self.maskedLayerNorm, vidinSize, dModel, dModel)
            videoEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
            self.videoEncoder = nn.TransformerEncoder(videoEncoderLayer, num_layers=numLayers, norm=tx_norm)
        else:
            self.visualModel = None
            self.videoConv = None
            self.videoEncoder = None
        # JointConv for fusion
        if self.modal == "AV":
            self.jointConv = conv1dLayers(self.maskedLayerNorm, 2 * dModel, dModel, dModel)
            jointEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
            self.jointEncoder = nn.TransformerEncoder(jointEncoderLayer, num_layers=numLayers, norm=tx_norm)
        self.jointOutputConv = outputConv(self.maskedLayerNorm, dModel, numClasses)
        self.decoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        self.embed = torch.nn.Sequential(
            nn.Embedding(numClasses, dModel),
            self.decoderPositionalEncoding
        )
        jointDecoderLayer = nn.TransformerDecoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.jointAttentionDecoder = nn.TransformerDecoder(jointDecoderLayer, num_layers=numLayers, norm=tx_norm)
        self.jointAttentionOutputConv = outputConv("LN", dModel, numClasses)

        #LM
        #path: /home/gryang/LM_results/LRS23vocab_LibriLRS23_wordpiece/all_checkpoints/drop0.5wd0.01/checkpoint1130.pt
        path="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/final_lm/checkpoint1130.pt"
        transformer_ckptprefix="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/final_lm/"
        transformer_ckpt="checkpoint1130.pt"
        self.transformer_lm=TransformerLanguageModel.from_pretrained(transformer_ckptprefix, transformer_ckpt)
        
        #path= /home/gryang/LM_results/LibriLRS23_wordpiecedata/LibriLRS23_vocab4000.json
        self.bert_tokenizer = Tokenizer.from_file("/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/final_lm/LibriLRS23_vocab4000.json")
        return

    def subNetForward(self, inputBatch, maskw2v):
        audioBatch, audMask, videoBatch, vidLen = inputBatch
        if not self.modal == "VO":
            result = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v)  #new_version
            audioBatch,audMask =result["x"],result["padding_mask"]  #torch.Size([12, 310, 1024]) torch.Size([12, 310])  #如果是old version 是tuple #result[0]=torch.Size([12, 310, 1024])  result[1]=torch.Size([12, 310]) 
            audLen = torch.sum(~audMask, dim=1)   #tensor([89], device='cuda:1') 
        else:
            audLen = None

        if not self.modal == "AO":
            videoBatch = videoBatch.transpose(1, 2)
            videoBatch = self.visualModel(videoBatch, vidLen.long())
            videoBatch = list(torch.split(videoBatch, vidLen.tolist(), dim=0))

        audioBatch, videoBatch, inputLenBatch, mask = self.makePadding(audioBatch, audLen, videoBatch, vidLen)

        if isinstance(self.maskedLayerNorm, MaskedLayerNorm):
            self.maskedLayerNorm.SetMaskandLength(mask, inputLenBatch)

        if not self.modal == "VO":
            audioBatch = audioBatch.transpose(1, 2)
            audioBatch = self.audioConv(audioBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            audioBatch = self.EncoderPositionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask)

        if not self.modal == "AO":
            videoBatch = videoBatch.transpose(1, 2)
            videoBatch = self.videoConv(videoBatch)
            videoBatch = videoBatch.transpose(1, 2).transpose(0, 1)
            videoBatch = self.EncoderPositionalEncoding(videoBatch)
            videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask)

        if self.modal == "AO":
            jointBatch = audioBatch
        elif self.modal == "VO":
            jointBatch = videoBatch
        else:
            jointBatch = torch.cat([self.ModalityNormalization(audioBatch), self.ModalityNormalization(videoBatch)], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
            jointBatch = self.EncoderPositionalEncoding(jointBatch)
            jointBatch = self.jointEncoder(jointBatch, src_key_padding_mask=mask)

        return jointBatch, inputLenBatch, mask

    def forward(self, inputBatch, targetinBatch, targetLenBatch, maskw2v):
        jointBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v)
        jointCTCOutputBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointCTCOutputBatch = self.jointOutputConv(jointCTCOutputBatch)
        jointCTCOutputBatch = jointCTCOutputBatch.transpose(1, 2).transpose(0, 1)
        jointCTCOutputBatch = F.log_softmax(jointCTCOutputBatch, dim=2)

        targetinBatch = self.embed(targetinBatch.transpose(0, 1))
        targetinMask = self.makeMaskfromLength(targetinBatch.shape[:-1][::-1], targetLenBatch, targetinBatch.device)
        squareMask = generate_square_subsequent_mask(targetinBatch.shape[0], targetinBatch.device)
        jointAttentionOutputBatch = self.jointAttentionDecoder(targetinBatch, jointBatch, tgt_mask=squareMask,
                                                               tgt_key_padding_mask=targetinMask, memory_key_padding_mask=mask)
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(0, 1).transpose(1, 2)
        jointAttentionOutputBatch = self.jointAttentionOutputConv(jointAttentionOutputBatch)
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(1, 2)

        outputBatch = (jointCTCOutputBatch, jointAttentionOutputBatch)
        return inputLenBatch, outputBatch

    def inference(self, inputBatch, maskw2v, device, Lambda, beamWidth, eosIx, blank):  #maskwv2=False, Lambda:0.1 ,beamwidth=5, blank=0, eosIx=39
        encodedBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v) #(80,1,512), [80] mask:(1,80) 全false
        CTCOutputConv = self.jointOutputConv  #输入 jointCTCOutputBatch
        attentionDecoder = self.jointAttentionDecoder    #输入targetinBatch ...
        attentionOutputConv = self.jointAttentionOutputConv  #输入 jointAttentionOutputBatch

        CTCOutputBatch = encodedBatch.transpose(0, 1).transpose(1, 2) #（1，512，80）
        CTCOutputBatch = CTCOutputConv(CTCOutputBatch)
        CTCOutputBatch = CTCOutputBatch.transpose(1, 2)  #（1，80，40）
        # claim batch and time step
        batch = CTCOutputBatch.shape[0] #1
        T = inputLenBatch.cpu()  # tensor([80])
        # claim CTClogprobs and Length
        CTCOutputBatch = CTCOutputBatch.cpu()
        CTCOutLogProbs = F.log_softmax(CTCOutputBatch, dim=-1) #（1，80，40） #!!!!
        predictionLenBatch = torch.ones(batch, device=device).long() #[1]
        # init Omega and Omegahat for attention beam search
        Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in range(batch)]  #[[[(tensor([39]), tensor(0), tensor(0))]]]
        Omegahat = [[] for i in range(batch)]  # [[]]
        # init
        gamma_n = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        gamma_b = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        for b in range(batch):
            gamma_b[b, 0, 0, 0] = 0
            for t in range(1, T[b]):
                gamma_n[b, t, 0, 0] = -np.inf
                gamma_b[b, t, 0, 0] = 0
                for tao in range(1, t + 1):
                    gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[b, tao, blank]   ###
        #这上面都是初始化
        newhypo = torch.arange(1, self.numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0) #(1,1,39,1)  #值好像是1-39

        for l in tqdm(range(1, T.max() + 1), leave=False, desc="Regression", ncols=75):
            predictionBatch = []
            for i in range(batch):
                predictionBatch += [x[0] for x in Omega[i][-1][:beamWidth]]  #[tensor([39])]
                Omega[i].append([])
            predictionBatch = torch.stack(predictionBatch).long().to(device)    #tensor([[39]]） 但是我疑惑为什么是eos
            predictionBatch = self.embed(predictionBatch.transpose(0, 1))  #(1,1,512)  给那个decode预测值编码 好输入decoder  #self.embed = torch.nn.Sequential(nn.Embedding(numClasses, dModel),self.decoderPositionalEncoding)
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool() #tensor([[False]]
            if not predictionBatch.shape[1] == encodedBatch.shape[1]:  #x
                encoderIndex = [i for i in range(batch) for j in range(beamWidth)]
                encodedBatch = encodedBatch[:, encoderIndex, :]
                mask = mask[encoderIndex]
                predictionLenBatch = predictionLenBatch[encoderIndex]
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)  #（1，1，512） ？
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)  # (1,512,1)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)  #(1,80,40)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1) #(1,39) 这里取了-1 和 1
            attentionOutLogProbs = attentionOutputBatch.unsqueeze(1).cpu() #(1,1,39)

            # Decode
            h = []
            alpha = []
            for b in range(batch):
                h.append([])
                alpha.append([])
                for o in Omega[b][l - 1][:beamWidth]:
                    h[b].append([o[0].tolist()])
                    alpha[b].append([[o[1], o[2]]])
            h = torch.tensor(h)
            alpha = torch.tensor(alpha).float()
            numBeam = alpha.shape[1]
            recurrnewhypo = torch.repeat_interleave(torch.repeat_interleave(newhypo, batch, dim=0), numBeam, dim=1) #（1,1,39,1)
            h = torch.cat((torch.repeat_interleave(h, self.numClasses - 1, dim=2), recurrnewhypo), dim=-1)  #(1,1,39,2) [39,1],[39,2],[39,3],...   #第二轮(1,5,39,3)  t=2时选的5个是20，5，9，11，13 [39,20,1][39,20,2]...  #第三轮(1,5,39,4) [39,20, 13/2/8/20/16, 1-39] check了每一轮的h就是这个

            alpha = torch.repeat_interleave(alpha, self.numClasses - 1, dim=2)
            alpha[:, :, :, 1] += attentionOutLogProbs.reshape(batch, numBeam, -1) #第二项的值都是attentionoutlogprobs里的 #（1,1,39,2)[ 0.0000e+00, -9.2155e+00],[ 0.0000e+00, -8.9963e+00],类似这样第一项都是0

            # h = (batch * beam * 39 * hypoLength)
            # alpha = (batch * beam * 39)
            # CTCOutLogProbs = (batch * sequence length * 40)    #第一次 alpha[:, :, :, 1]torch.Size([1, 1, 39])==attentionOutLogProbs
            # gamma_n or gamma_b = (batch * max time length * beamwidth * 40 <which is max num of candidates in one time step>)
            CTCHypoLogProbs = compute_CTC_prob(h, alpha[:, :, :, 1], CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, self.numClasses - 1, blank, eosIx)
            alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]  #(1,1,39,2)  #直接在这加就可以了  torch.Size([1, 5, 39])
            hPaddingShape = list(h.shape)  #(1,1,39,2)
            hPaddingShape[-2] = 1  #(1,1,1,2)
            h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)  #(1,1,40,2) [[0,0],[39,1],[39,2],[39,3],...]
            activeBatch = (l < T).nonzero().squeeze(-1).tolist()
            for b in activeBatch:
                for i in range(numBeam):
                    Omegahat[b].append((h[b, i, -1], alpha[b, i, -1, 0]))

            alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)
            alpha[:, :, -1, 0] = -np.inf
            predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices
            for b in range(batch):
                for pos, c in enumerate(predictionRes[b]):
                    beam = c // self.numClasses
                    c = c % self.numClasses
                    Omega[b][l].append((h[b, beam, c], alpha[b, beam, c, 0], alpha[b, beam, c, 1]))
                    gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                    gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
            gamma_n[:, :, :, 1:] = -np.inf
            gamma_b[:, :, :, 1:] = -np.inf
            predictionLenBatch += 1

        predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]   #[tensor,tensor]
        predictionLenBatch = [len(prediction) - 1 for prediction in predictionBatch]
        return torch.cat([prediction[1:] for prediction in predictionBatch]).int(),   torch.tensor(predictionLenBatch).int()  #(tensor([20, 13,  3,  1,  2, 23,  2,  7,  3, 13,  5, 11, 11, 14,  1,  3,  9,  2,14,  1, 12,  6, 12,  1, 17,  4, 18,  2,  1,  5, 10,  4, 13,  7, 12, 39],dtype=torch.int32), tensor(36, dtype=torch.int32))

    def rescore(self, inputBatch, maskw2v, device, Lambda, beamWidth, eosIx, blank, beta, nbest,logger):
        self.transformer_lm.to(device)
        
        encodedBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v) #(80,1,512), [80] mask:(1,80) 全false
        CTCOutputConv = self.jointOutputConv  #输入 jointCTCOutputBatch
        attentionDecoder = self.jointAttentionDecoder    #输入targetinBatch ...
        attentionOutputConv = self.jointAttentionOutputConv  #输入 jointAttentionOutputBatch

        CTCOutputBatch = encodedBatch.transpose(0, 1).transpose(1, 2) #（1，512，80）
        CTCOutputBatch = CTCOutputConv(CTCOutputBatch)
        CTCOutputBatch = CTCOutputBatch.transpose(1, 2)  #（1，80，40）
        # claim batch and time step
        batch = CTCOutputBatch.shape[0] #1
        T = inputLenBatch.cpu()  # tensor([80])
        # claim CTClogprobs and Length
        CTCOutputBatch = CTCOutputBatch.cpu()
        CTCOutLogProbs = F.log_softmax(CTCOutputBatch, dim=-1) #（1，80，40） #!!!!
        predictionLenBatch = torch.ones(batch, device=device).long() #[1]
        # init Omega and Omegahat for attention beam search
        Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in range(batch)]  #[[[(tensor([39]), tensor(0), tensor(0))]]]
        Omegahat = [[] for i in range(batch)]  # [[]]
        # init
        gamma_n = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        gamma_b = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        for b in range(batch):
            gamma_b[b, 0, 0, 0] = 0
            for t in range(1, T[b]):
                gamma_n[b, t, 0, 0] = -np.inf
                gamma_b[b, t, 0, 0] = 0
                for tao in range(1, t + 1):
                    gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[b, tao, blank]   ###
        #这上面都是初始化
        newhypo = torch.arange(1, self.numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0) #(1,1,39,1)  #值好像是1-39

        for l in tqdm(range(1, T.max() + 1), leave=False, desc="Regression", ncols=75):
            predictionBatch = []
            for i in range(batch):
                predictionBatch += [x[0] for x in Omega[i][-1][:beamWidth]]  #[tensor([39])]
                Omega[i].append([])
            predictionBatch = torch.stack(predictionBatch).long().to(device)    #tensor([[39]]） 但是我疑惑为什么是eos
            predictionBatch = self.embed(predictionBatch.transpose(0, 1))  #(1,1,512)  给那个decode预测值编码 好输入decoder  #self.embed = torch.nn.Sequential(nn.Embedding(numClasses, dModel),self.decoderPositionalEncoding)
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool() #tensor([[False]]
            if not predictionBatch.shape[1] == encodedBatch.shape[1]:  #x
                encoderIndex = [i for i in range(batch) for j in range(beamWidth)]
                encodedBatch = encodedBatch[:, encoderIndex, :]
                mask = mask[encoderIndex]
                predictionLenBatch = predictionLenBatch[encoderIndex]
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)  #（1，1，512） ？
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)  # (1,512,1)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)  #(1,80,40)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1) #(1,39) 这里取了-1 和 1
            attentionOutLogProbs = attentionOutputBatch.unsqueeze(1).cpu() #(1,1,39)

            # Decode
            h = []
            alpha = []
            for b in range(batch):
                h.append([])
                alpha.append([])
                for o in Omega[b][l - 1][:beamWidth]:
                    h[b].append([o[0].tolist()])
                    alpha[b].append([[o[1], o[2]]])
            h = torch.tensor(h)
            alpha = torch.tensor(alpha).float()
            numBeam = alpha.shape[1]
            recurrnewhypo = torch.repeat_interleave(torch.repeat_interleave(newhypo, batch, dim=0), numBeam, dim=1) #（1,1,39,1)
            h = torch.cat((torch.repeat_interleave(h, self.numClasses - 1, dim=2), recurrnewhypo), dim=-1)  #(1,1,39,2) [39,1],[39,2],[39,3],...   #第二轮(1,5,39,3)  t=2时选的5个是20，5，9，11，13 [39,20,1][39,20,2]...  #第三轮(1,5,39,4) [39,20, 13/2/8/20/16, 1-39] check了每一轮的h就是这个

            alpha = torch.repeat_interleave(alpha, self.numClasses - 1, dim=2)
            alpha[:, :, :, 1] += attentionOutLogProbs.reshape(batch, numBeam, -1) #第二项的值都是attentionoutlogprobs里的 #（1,1,39,2)[ 0.0000e+00, -9.2155e+00],[ 0.0000e+00, -8.9963e+00],类似这样第一项都是0

            # h = (batch * beam * 39 * hypoLength)
            # alpha = (batch * beam * 39)
            # CTCOutLogProbs = (batch * sequence length * 40)    #第一次 alpha[:, :, :, 1]torch.Size([1, 1, 39])==attentionOutLogProbs
            # gamma_n or gamma_b = (batch * max time length * beamwidth * 40 <which is max num of candidates in one time step>)
            CTCHypoLogProbs = compute_CTC_prob(h, alpha[:, :, :, 1], CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, self.numClasses - 1, blank, eosIx)
            alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]  #(1,1,39,2)  #直接在这加就可以了  torch.Size([1, 5, 39])
            hPaddingShape = list(h.shape)  #(1,1,39,2)
            hPaddingShape[-2] = 1  #(1,1,1,2)
            h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)  #(1,1,40,2) [[0,0],[39,1],[39,2],[39,3],...]
            activeBatch = (l < T).nonzero().squeeze(-1).tolist()
            for b in activeBatch:
                for i in range(numBeam):
                    Omegahat[b].append((h[b, i, -1], alpha[b, i, -1, 0]))

            alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)
            alpha[:, :, -1, 0] = -np.inf
            predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices
            for b in range(batch):
                for pos, c in enumerate(predictionRes[b]):
                    beam = c // self.numClasses
                    c = c % self.numClasses
                    Omega[b][l].append((h[b, beam, c], alpha[b, beam, c, 0], alpha[b, beam, c, 1]))
                    gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                    gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
            gamma_n[:, :, :, 1:] = -np.inf
            gamma_b[:, :, :, 1:] = -np.inf
            predictionLenBatch += 1

        predictionBatchs = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[:nbest] for b in range(batch)] # 多batch版本 nbest=100 取前100个
        newpredictionBatchs=[]
        predictionLenBatchs= []
        for i in range(len(predictionBatchs)):
            predictionBatch= predictionBatchs[i]
            newpredictionBatch=[]
            for j in range(nbest):
                one= predictionBatch[j]
                sentence,acscore = predictionBatch[j][0][1:].int(),predictionBatch[j][1]  #senctence:tensor([20, 13,  3,  1,  2, 23,  2,  7,  3, 13,  5, 11, 11, 14,  1,  3,  9,  2,...] #-3.127924919128418
                lmscore= self.transformerscore(sentence,device)
                sumscore=acscore+ beta * lmscore
                newpredictionBatch.append((sumscore,j,sentence))
                #newpredictionBatch.append((sumscore,j,sentence,acscore,lmscore))
                
            sortedBatch = sorted(newpredictionBatch,key=lambda x: (-x[0],x[1]),reverse=False)  #第一个元素降序，第二个元素升序
            index=sortedBatch[0][1]
            newsenctence= sortedBatch[0][2]
            #acscore=sortedBatch[0][3]
            #lmscore=sortedBatch[0][4]
            #logger.info("sentence:%d,acscore:%f,lmscore:%f"%(i,acscore,lmscore))
            if index!=0:
                logger.info("rescore index:"+str(index))
            newpredictionBatchs.append(newsenctence)
            predictionLenBatchs.append(len(newsenctence))

        newpredictionBatchs=torch.cat(newpredictionBatchs)
        
        return newpredictionBatchs, torch.tensor(predictionLenBatchs).int()    
    
    def transformerscore(self,sentence,device): #ngram  #sentence是一个tensor
        dict=args["INDEX_TO_CHAR"]  
        one_sentence = sentence.tolist()  # [39, 20, 13, 39]
        out = itemgetter(*one_sentence)(dict) 
        out = out[:-1]  
        out = ''.join(out)  # 'BUT EVENTUALLY THEY DID COME AROUND' #out=out.to(device)

        # sentencepiece encoder
        # encoder= self.sentencepieceprocessor.encode(out,out_type=str)  #['▁BUT', '▁EVENTUALLY', '▁THEY', '▁DID', '▁COME', '▁AROUND'] new!!!
        # bpe_result=" ".join(encoder)   #'▁BUT ▁EVENTUALLY ▁THEY ▁DID ▁COME ▁AROUND'
        
        # wordpiece encoder
        output= self.bert_tokenizer.encode(out) #print(output.tokens)
        output=output.tokens[1:-1] #['PROFESSION', '##AL', 'THEY', 'DID', 'COME', 'AROUND']
        bpe_result=" ".join(output)   #'PROFESSION ##AL THEY DID COME AROUND'      #忘了删这行不小心多跑了一次就会变成  a b c d这样fuck
        
        scoreall = self.transformer_lm.score(bpe_result)  #这一步算的很快  #不需要取平均！LMscore=scoreall['score']  #tensor(-4.7877, device='cuda:0') 
        score = scoreall["positional_scores"].sum() #tensor([-3.6244, -8.2294, -2.7765, -4.0579, -3.8147, -5.9651], device='cuda:0') tensor(-28.4681, device='cuda:0')
        return score #tensor(-39.0554, device='cuda:0')
        
    def shallow_fusion(self, inputBatch, maskw2v, device, Lambda, beamWidth, eosIx, blank,beta):  #maskwv2=False, Lambda:0.1 ,beamwidth=5, blank=0, eosIx=39
        encodedBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v) #(80,1,512), [80] mask:(1,80) 全false
        CTCOutputConv = self.jointOutputConv  #输入 jointCTCOutputBatch
        attentionDecoder = self.jointAttentionDecoder    #输入targetinBatch ...
        attentionOutputConv = self.jointAttentionOutputConv  #输入 jointAttentionOutputBatch

        CTCOutputBatch = encodedBatch.transpose(0, 1).transpose(1, 2) #（1，512，80）
        CTCOutputBatch = CTCOutputConv(CTCOutputBatch)
        CTCOutputBatch = CTCOutputBatch.transpose(1, 2)  #（1，80，40）
        # claim batch and time step
        batch = CTCOutputBatch.shape[0] #1
        T = inputLenBatch.cpu()  # tensor([80])
        # claim CTClogprobs and Length
        CTCOutputBatch = CTCOutputBatch.cpu()
        CTCOutLogProbs = F.log_softmax(CTCOutputBatch, dim=-1) #（1，80，40） #!!!!
        predictionLenBatch = torch.ones(batch, device=device).long() #[1]
        # init Omega and Omegahat for attention beam search
        Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in range(batch)]  #[[[(tensor([39]), tensor(0), tensor(0))]]]
        Omegahat = [[] for i in range(batch)]  # [[]]
        # init
        gamma_n = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        gamma_b = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        for b in range(batch):
            gamma_b[b, 0, 0, 0] = 0
            for t in range(1, T[b]):
                gamma_n[b, t, 0, 0] = -np.inf
                gamma_b[b, t, 0, 0] = 0
                for tao in range(1, t + 1):
                    gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[b, tao, blank]   ###
        #这上面都是初始化
        newhypo = torch.arange(1, self.numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0) #(1,1,39,1)  #值好像是1-39

        for l in tqdm(range(1, T.max() + 1), leave=False, desc="Regression", ncols=75):
            predictionBatch = []
            for i in range(batch):
                predictionBatch += [x[0] for x in Omega[i][-1][:beamWidth]]  #[tensor([39])]
                Omega[i].append([])
            predictionBatch = torch.stack(predictionBatch).long().to(device)    #tensor([[39]]） 但是我疑惑为什么是eos
            predictionBatch = self.embed(predictionBatch.transpose(0, 1))  #(1,1,512)  给那个decode预测值编码 好输入decoder  #self.embed = torch.nn.Sequential(nn.Embedding(numClasses, dModel),self.decoderPositionalEncoding)
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool() #tensor([[False]]
            if not predictionBatch.shape[1] == encodedBatch.shape[1]:  #x
                encoderIndex = [i for i in range(batch) for j in range(beamWidth)]
                encodedBatch = encodedBatch[:, encoderIndex, :]
                mask = mask[encoderIndex]
                predictionLenBatch = predictionLenBatch[encoderIndex]
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)  #（1，1，512） ？
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)  # (1,512,1)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)  #(1,80,40)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1) #(1,39) 这里取了-1 和 1
            attentionOutLogProbs = attentionOutputBatch.unsqueeze(1).cpu() #(1,1,39)

            # Decode
            h = []
            alpha = []
            for b in range(batch):
                h.append([])
                alpha.append([])
                for o in Omega[b][l - 1][:beamWidth]:
                    h[b].append([o[0].tolist()])
                    alpha[b].append([[o[1], o[2]]])
            h = torch.tensor(h)
            alpha = torch.tensor(alpha).float()
            numBeam = alpha.shape[1]
            recurrnewhypo = torch.repeat_interleave(torch.repeat_interleave(newhypo, batch, dim=0), numBeam, dim=1) #（1,1,39,1)
            h = torch.cat((torch.repeat_interleave(h, self.numClasses - 1, dim=2), recurrnewhypo), dim=-1)  #(1,1,39,2) [39,1],[39,2],[39,3],...   #第二轮(1,5,39,3)  t=2时选的5个是20，5，9，11，13 [39,20,1][39,20,2]...  #第三轮(1,5,39,4) [39,20, 13/2/8/20/16, 1-39] check了每一轮的h就是这个

            alpha = torch.repeat_interleave(alpha, self.numClasses - 1, dim=2)
            alpha[:, :, :, 1] += attentionOutLogProbs.reshape(batch, numBeam, -1) #第二项的值都是attentionoutlogprobs里的 #（1,1,39,2)[ 0.0000e+00, -9.2155e+00],[ 0.0000e+00, -8.9963e+00],类似这样第一项都是0

            # h = (batch * beam * 39 * hypoLength)
            # alpha = (batch * beam * 39)
            # CTCOutLogProbs = (batch * sequence length * 40)    #第一次 alpha[:, :, :, 1]torch.Size([1, 1, 39])==attentionOutLogProbs
            # gamma_n or gamma_b = (batch * max time length * beamwidth * 40 <which is max num of candidates in one time step>)


            CTCHypoLogProbs = compute_CTC_prob(h, alpha[:, :, :, 1], CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, self.numClasses - 1, blank, eosIx)
            alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]#  #(1,1,39,2)  #直接在这加就可以了
            
            if (h.shape)[-1]>=3:  #至少有一个字母之后加上LM
                #word_boundary
                sentence_eos=h[:,:,-1,:][0]  #torch.Size([5, 4])  第一个和后面的shape不一样
                sentence_kongge=h[:,:,0,:][0] 
                
                twoscore=self.beam_transformerscore(sentence_eos,device)
                LmLogProbs_eos= twoscore
                LmLogProbs_kongge= twoscore  #只是换了个语言模型
                alpha[:, :, -1, 0] += beta * LmLogProbs_eos  
                alpha[:, :, 0, 0] += beta * LmLogProbs_kongge  

            hPaddingShape = list(h.shape)  #(1,1,39,2)
            hPaddingShape[-2] = 1  #(1,1,1,2)
            h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)  #(1,1,40,2) [[0,0],[39,1],[39,2],[39,3],...]
            activeBatch = (l < T).nonzero().squeeze(-1).tolist()
            for b in activeBatch:
                for i in range(numBeam):
                    Omegahat[b].append((h[b, i, -1], alpha[b, i, -1, 0]))

            alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)
            alpha[:, :, -1, 0] = -np.inf
            predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices
            for b in range(batch):
                for pos, c in enumerate(predictionRes[b]):
                    beam = c // self.numClasses
                    c = c % self.numClasses
                    Omega[b][l].append((h[b, beam, c], alpha[b, beam, c, 0], alpha[b, beam, c, 1]))
                    gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                    gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
            gamma_n[:, :, :, 1:] = -np.inf
            gamma_b[:, :, :, 1:] = -np.inf
            predictionLenBatch += 1

        predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]  #(80,5,512)   #b=0 len(Omega[0]):81
        predictionLenBatch = [len(prediction) - 1 for prediction in predictionBatch]
        return torch.cat([prediction[1:] for prediction in predictionBatch]).int(), torch.tensor(predictionLenBatch).int()

    def beam_transformerscore(self,sentence,device): #ngram  #sentence是一个tensor
        # tensor([[39., 20., 13., 39.],
        #         [39., 20., 2., 39.],
        #         [39., 20., 8., 39.],
        #         [39., 20., 20., 39.],
        #         [39., 20., 16., 39.]])   5,4
        
        dict=args["INDEX_TO_CHAR"]
        nsentence, length = sentence.shape[0], sentence.shape[1]  # 5,4
        result = []
        for i in range(nsentence):
            one_sentence = sentence[i].int().tolist()  # [39, 20, 13, 39]
            out = itemgetter(*one_sentence)(dict)  # ('<EOS>', 'B', 'U', '<EOS>')  #assert (out[0] == '<EOS>' and out[-1] == '<EOS>')  如果是空格就是  ('<EOS>', 'B', 'U', ' ')  
            out = out[1:-1]  # BU
            out = ''.join(out)  # print(out) 'BU'  #n-gram 大写就可以
            
            #sentencepiece 
            # encoder= self.sentencepieceprocessor.encode(out,out_type=str)  #new!!! ['▁B']
            # bpe_result=" ".join(encoder)    #'▁B'
            
            # wordpiece encoder
            output= self.bert_tokenizer.encode(out) #print(output.tokens)
            output=output.tokens[1:-1] #['PROFESSION', '##AL', 'THEY', 'DID', 'COME', 'AROUND']
            bpe_result=" ".join(output)   #'PROFESSION ##AL THEY DID COME AROUND'      #忘了删这行不小心多跑了一次就会变成  a b c d这样fuck
            
            scoreall = self.transformer_lm.score(bpe_result)  #这一步算的很快
            LMscore = scoreall["positional_scores"].sum()
            result.append(LMscore)  #-20.80895

        result_tensor = torch.tensor(result)
        return result_tensor  
    
    def oracle(self, inputBatch, maskw2v, device, Lambda, beamWidth, eosIx, blank): 
        encodedBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v) #(80,1,512), [80] mask:(1,80) 全false
        CTCOutputConv = self.jointOutputConv  #输入 jointCTCOutputBatch
        attentionDecoder = self.jointAttentionDecoder    #输入targetinBatch ...
        attentionOutputConv = self.jointAttentionOutputConv  #输入 jointAttentionOutputBatch

        CTCOutputBatch = encodedBatch.transpose(0, 1).transpose(1, 2) #（1，512，80）
        CTCOutputBatch = CTCOutputConv(CTCOutputBatch)
        CTCOutputBatch = CTCOutputBatch.transpose(1, 2)  #（1，80，40）
        # claim batch and time step
        batch = CTCOutputBatch.shape[0] #1
        T = inputLenBatch.cpu()  # tensor([80])
        # claim CTClogprobs and Length
        CTCOutputBatch = CTCOutputBatch.cpu()
        CTCOutLogProbs = F.log_softmax(CTCOutputBatch, dim=-1) #（1，80，40） #!!!!
        predictionLenBatch = torch.ones(batch, device=device).long() #[1]
        # init Omega and Omegahat for attention beam search
        Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in range(batch)]  #[[[(tensor([39]), tensor(0), tensor(0))]]]
        Omegahat = [[] for i in range(batch)]  # [[]]
        # init
        gamma_n = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        gamma_b = torch.full((batch, T.max(), beamWidth, self.numClasses), -np.inf).float()  #(1,80,5,40) -np.inf
        for b in range(batch):
            gamma_b[b, 0, 0, 0] = 0
            for t in range(1, T[b]):
                gamma_n[b, t, 0, 0] = -np.inf
                gamma_b[b, t, 0, 0] = 0
                for tao in range(1, t + 1):
                    gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[b, tao, blank]   ###
        #这上面都是初始化
        newhypo = torch.arange(1, self.numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0) #(1,1,39,1)  #值好像是1-39

        for l in tqdm(range(1, T.max() + 1), leave=False, desc="Regression", ncols=75):
            predictionBatch = []
            for i in range(batch):
                predictionBatch += [x[0] for x in Omega[i][-1][:beamWidth]]  #[tensor([39])]
                Omega[i].append([])
            predictionBatch = torch.stack(predictionBatch).long().to(device)    #tensor([[39]]） 但是我疑惑为什么是eos
            predictionBatch = self.embed(predictionBatch.transpose(0, 1))  #(1,1,512)  给那个decode预测值编码 好输入decoder  #self.embed = torch.nn.Sequential(nn.Embedding(numClasses, dModel),self.decoderPositionalEncoding)
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool() #tensor([[False]]
            if not predictionBatch.shape[1] == encodedBatch.shape[1]:  #x
                encoderIndex = [i for i in range(batch) for j in range(beamWidth)]
                encodedBatch = encodedBatch[:, encoderIndex, :]
                mask = mask[encoderIndex]
                predictionLenBatch = predictionLenBatch[encoderIndex]
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)  #（1，1，512） ？
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)  # (1,512,1)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)  #(1,80,40)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1) #(1,39) 这里取了-1 和 1
            attentionOutLogProbs = attentionOutputBatch.unsqueeze(1).cpu() #(1,1,39)

            # Decode
            h = []
            alpha = []
            for b in range(batch):
                h.append([])
                alpha.append([])
                for o in Omega[b][l - 1][:beamWidth]:
                    h[b].append([o[0].tolist()])
                    alpha[b].append([[o[1], o[2]]])
            h = torch.tensor(h)
            alpha = torch.tensor(alpha).float()
            numBeam = alpha.shape[1]
            recurrnewhypo = torch.repeat_interleave(torch.repeat_interleave(newhypo, batch, dim=0), numBeam, dim=1) #（1,1,39,1)
            h = torch.cat((torch.repeat_interleave(h, self.numClasses - 1, dim=2), recurrnewhypo), dim=-1)  #(1,1,39,2) [39,1],[39,2],[39,3],...   #第二轮(1,5,39,3)  t=2时选的5个是20，5，9，11，13 [39,20,1][39,20,2]...  #第三轮(1,5,39,4) [39,20, 13/2/8/20/16, 1-39] check了每一轮的h就是这个

            alpha = torch.repeat_interleave(alpha, self.numClasses - 1, dim=2)
            alpha[:, :, :, 1] += attentionOutLogProbs.reshape(batch, numBeam, -1) #第二项的值都是attentionoutlogprobs里的 #（1,1,39,2)[ 0.0000e+00, -9.2155e+00],[ 0.0000e+00, -8.9963e+00],类似这样第一项都是0

            # h = (batch * beam * 39 * hypoLength)
            # alpha = (batch * beam * 39)
            # CTCOutLogProbs = (batch * sequence length * 40)    #第一次 alpha[:, :, :, 1]torch.Size([1, 1, 39])==attentionOutLogProbs
            # gamma_n or gamma_b = (batch * max time length * beamwidth * 40 <which is max num of candidates in one time step>)
            CTCHypoLogProbs = compute_CTC_prob(h, alpha[:, :, :, 1], CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, self.numClasses - 1, blank, eosIx)
            alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]  #(1,1,39,2)  #直接在这加就可以了  torch.Size([1, 5, 39])
            hPaddingShape = list(h.shape)  #(1,1,39,2)
            hPaddingShape[-2] = 1  #(1,1,1,2)
            h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)  #(1,1,40,2) [[0,0],[39,1],[39,2],[39,3],...]
            activeBatch = (l < T).nonzero().squeeze(-1).tolist()
            for b in activeBatch:
                for i in range(numBeam):
                    Omegahat[b].append((h[b, i, -1], alpha[b, i, -1, 0]))

            alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)
            alpha[:, :, -1, 0] = -np.inf
            predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices
            for b in range(batch):
                for pos, c in enumerate(predictionRes[b]):
                    beam = c // self.numClasses
                    c = c % self.numClasses
                    Omega[b][l].append((h[b, beam, c], alpha[b, beam, c, 0], alpha[b, beam, c, 1]))
                    gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                    gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
            gamma_n[:, :, :, 1:] = -np.inf
            gamma_b[:, :, :, 1:] = -np.inf
            predictionLenBatch += 1

        predictionBatchs = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[:100] for b in range(batch)] # 多batch版本 nbest=100 取前100个
        predictionLenBatchs= []
        for i in range(len(predictionBatchs)):
            predictionBatch= predictionBatchs[i]
            predictionBatch= [i[0][1:].int() for i in predictionBatch ]  #去掉bos 39
            predictionBatchs[i] = predictionBatch
            
            predictionLenBatch = [len(prediction) for prediction in predictionBatch]
            predictionLenBatchs.append(predictionLenBatch)
        
        return predictionBatchs, torch.tensor(predictionLenBatchs).int()     
    
    def ngram_lmscore(self,sentence,device):  #ngram
        dict=args["INDEX_TO_CHAR"]
        nsentence, length = sentence.shape[0], sentence.shape[1]  # 5,4
        result = []
        for i in range(nsentence):
            one_sentence = sentence[i].int().tolist()  # [39, 20, 13, 39]
            out = itemgetter(*one_sentence)(dict)  # ('<EOS>', 'B', 'U', '<EOS>')  
            out = out[1:-1]  # BU
            out = ''.join(out)  # print(out) 'BU'  #n-gram 大写就可以

            # logsoftmax 以e为底，kenlm 以10为底，所以要换一下 
            LMscore= (self.ngram).score(out, bos=True, eos=True)   #-9.037 #Return the log10 probability of a string.  By default, the string is treated as a sentence. return log10 p(sentence </s> | <s>)
            LMscore= LMscore/ ( math.log10(math.e) )  # 分母是0.4342944819032518

            result.append(LMscore)  #-20.80895

        result_tensor = torch.tensor(result)
        return result_tensor 
            
    def huggingface_lmscore(self,sentence,device):
        # tensor([[39., 20., 13., 39.],
        #         [39., 20., 2., 39.],
        #         [39., 20., 8., 39.],
        #         [39., 20., 20., 39.],
        #         [39., 20., 16., 39.]])   5,4

        dict=args["INDEX_TO_CHAR"]
        nsentence, length = sentence.shape[0], sentence.shape[1]  # 5,4
        result = []
        for i in range(nsentence):
            one_sentence = sentence[i].int().tolist()  # [39, 20, 13, 39]
            out = itemgetter(*one_sentence)(dict)  # ('<EOS>', 'B', 'U', '<EOS>')
            assert (out[0] == '<EOS>' and out[-1] == '<EOS>')
            out = out[1:-1]  # BU
            out = ''.join(out)  # print(out) 'BU'
            lower = out.lower()  # print(out.lower()) 'bu'
            cap = lower.capitalize()  # print(cap)  'Bu'
            cap = cap + '.'  # 一个词用那个LM 会麻烦  'Bu.'

            inputs = self.tokenizer(cap,return_tensors="pt")  # {'input_ids': tensor([[51827,3]])}   #inputs["input_ids"][0]  tensor([68, 2271, 3463])
            inputs.to(device)
            outputs = self.LM(**inputs, labels=inputs["input_ids"])
            losses = outputs.losses  # [4.5008, 8.4680] Negative log likelihood   [(len-1)*bsz] Negative log likelihood.
            logp = -losses.sum()
            result.append(logp)

        result_tensor = torch.tensor(result)
        return result_tensor  

    def attentionAutoregression(self, inputBatch, maskw2v, device, eosIx):
        encodedBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v)
        attentionDecoder = self.jointAttentionDecoder
        attentionOutputConv = self.jointAttentionOutputConv

        # claim batch and time step
        batch = encodedBatch.shape[1]
        T = inputLenBatch.cpu()
        # claim CTClogprobs and Length
        predictionLenBatch = torch.ones(batch, device=device).long()
        endMask = torch.ones(batch, device=device).bool()
        predictionInpBatch = torch.full((batch, 1), eosIx, device=device).long()

        while endMask.max() and predictionLenBatch.max() < T.max():
            predictionBatch = self.embed(predictionInpBatch.transpose(0, 1))
            targetinMask = torch.zeros(predictionBatch.shape[:-1][::-1], device=device).bool()
            squareMask = generate_square_subsequent_mask(predictionBatch.shape[0], device)
            attentionOutputBatch = attentionDecoder(predictionBatch, encodedBatch, tgt_mask=squareMask, tgt_key_padding_mask=targetinMask,
                                                    memory_key_padding_mask=mask)
            attentionOutputBatch = attentionOutputBatch.transpose(0, 1).transpose(1, 2)
            attentionOutputBatch = attentionOutputConv(attentionOutputBatch)
            attentionOutputBatch = attentionOutputBatch.transpose(1, 2)
            attentionOutputBatch = F.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1)
            predictionNewBatch = torch.argmax(attentionOutputBatch, dim=-1) + 1
            endMask *= ~(predictionNewBatch == eosIx)
            predictionNewBatch = predictionNewBatch.unsqueeze(0).transpose(0, 1)
            predictionInpBatch = torch.cat((predictionInpBatch, predictionNewBatch), dim=-1)
            predictionLenBatch[endMask] += 1
        predictionInpBatch = torch.cat((predictionInpBatch, torch.full((batch, 1), eosIx, device=device)), dim=-1)
        return torch.cat([predictionInp[1:predictionLenBatch[b] + 1] for b, predictionInp in enumerate(predictionInpBatch)]).int().cpu(), \
               predictionLenBatch.int().cpu()

    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask

    def makePadding(self, audioBatch, audLen, videoBatch, vidLen):
        if self.modal == "AO":
            audPadding = audLen % 2
            mask = (audPadding + audLen) > 2 * self.reqInpLen
            audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)
            audLeftPadding = torch.floor(torch.div(audPadding, 2)).int()
            audRightPadding = torch.ceil(torch.div(audPadding, 2)).int()

            audioBatch = audioBatch.unsqueeze(1).unsqueeze(1)
            audioBatch = list(audioBatch)
            for i, _ in enumerate(audioBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding[i], audRightPadding[i]))
                audioBatch[i] = pad(audioBatch[i][:, :, :audLen[i]]).squeeze(0).squeeze(0)

            audioBatch = pad_sequence(audioBatch, batch_first=True)
            inputLenBatch = ((audLen + audPadding) // 2).long()
            mask = self.makeMaskfromLength([audioBatch.shape[0]] + [audioBatch.shape[1] // 2], inputLenBatch, audioBatch.device)

        elif self.modal == "VO":
            vidPadding = torch.zeros(len(videoBatch)).long().to(vidLen.device)

            mask = (vidPadding + vidLen) > self.reqInpLen
            vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen)

            vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
            vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()

            for i, _ in enumerate(videoBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
                videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            videoBatch = pad_sequence(videoBatch, batch_first=True)
            inputLenBatch = (vidLen + vidPadding).long()
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, videoBatch.device)

        else:
            dismatch = audLen - 2 * vidLen
            vidPadding = torch.ceil(torch.div(dismatch, 2)).int()
            vidPadding = vidPadding * (vidPadding > 0)
            audPadding = 2 * vidPadding - dismatch

            mask = (vidPadding + vidLen) > self.reqInpLen
            vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen)
            mask = (audPadding + audLen) > 2 * self.reqInpLen
            audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)

            vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
            vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()
            audLeftPadding = torch.floor(torch.div(audPadding, 2)).int()
            audRightPadding = torch.ceil(torch.div(audPadding, 2)).int()

            audioBatch = audioBatch.unsqueeze(1).unsqueeze(1)
            audioBatch = list(audioBatch)
            for i, _ in enumerate(audioBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding[i], audRightPadding[i]))
                audioBatch[i] = pad(audioBatch[i][:, :, :audLen[i]]).squeeze(0).squeeze(0)
                pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
                videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            audioBatch = pad_sequence(audioBatch, batch_first=True)
            videoBatch = pad_sequence(videoBatch, batch_first=True)
            inputLenBatch = (vidLen + vidPadding).long()
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, videoBatch.device)

        return audioBatch, videoBatch, inputLenBatch, mask
