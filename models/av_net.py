import fairseq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.decoders import compute_CTC_prob
from utils.dictionary import Dictionary

from .moco_visual_frontend import MoCoVisualFrontend
from .utils import PositionalEncoding, conv1dLayers, outputConv, MaskedLayerNorm, generate_square_subsequent_mask

#from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from config import args
from operator import itemgetter

class AVNet(nn.Module):   #eval.py 会用这个

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
            wav2vecModel, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([W2Vfile], arg_overrides={
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

        # self.tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        # self.LM = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")   #这个有什么说法吗

        return

    def subNetForward(self, inputBatch, maskw2v):
        audioBatch, audMask, videoBatch, vidLen = inputBatch
        if not self.modal == "VO":   #新老版本的区别 0.12.2 是返回四个     return {"x": x,"padding_mask": padding_mask,"features": unmasked_features,"layer_results": layer_results,}   # v 0.10.2 是  if features_only:return {"x": x, "padding_mask": padding_mask}
            #result = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v)  #new_version
            #audioBatch,audMask =result["x"],result["padding_mask"]  #torch.Size([12, 310, 1024]) torch.Size([12, 310])  #如果是old version 是tuple #result[0]=torch.Size([12, 310, 1024])  result[1]=torch.Size([12, 310]) 
            #audLen=torch.tensor([audioBatch.shape[1]],device=audioBatch.device).long() #这个只适用于batch=1的情况 因为mask=None 草 batch>1 就会有padding_mask了 
            
            audioBatch, audMask = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v) #torch.Size([1,89,1024]),#torch.Size([1,89])
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



    # def flash_infer(self, inputBatch, maskw2v, device, Lambda, beamWidth, eosIx, blank):
    #     encodedBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v) #(80,1,512), [80] mask:(1,80) 全false
    #     CTCOutputConv = self.jointOutputConv  #输入 jointCTCOutputBatch
    #     CTCOutputBatch = encodedBatch.transpose(0, 1).transpose(1, 2) #（1，512，80）
    #     CTCOutputBatch = CTCOutputConv(CTCOutputBatch)
    #     CTCOutputBatch = CTCOutputBatch.transpose(1, 2)  #（1，80，40）
        
    #     dictionary=Dictionary.load('dict.ltr.txt')
    #     exit()
        
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

        predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]
        predictionLenBatch = [len(prediction) - 1 for prediction in predictionBatch]
        return torch.cat([prediction[1:] for prediction in predictionBatch]).int(), torch.tensor(predictionLenBatch).int()
    


    def my_inference(self, inputBatch, maskw2v, device, Lambda, beamWidth, eosIx, blank,beta):  #maskwv2=False, Lambda:0.1 ,beamwidth=5, blank=0, eosIx=39
        # (self.tokenizer).to(device)
        #(self.LM).to(device)

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

            if (h.shape)[-1]>=4:
                sentence=h[:,:,-1,:][0]  #torch.Size([5, 4])  第一个和后面的shape不一样
                LmLogProbs=self.p(sentence,device)
                alpha[:, :, -1, 0] += beta * LmLogProbs

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

        predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]
        predictionLenBatch = [len(prediction) - 1 for prediction in predictionBatch]
        return torch.cat([prediction[1:] for prediction in predictionBatch]).int(), torch.tensor(predictionLenBatch).int()

    def p(self,sentence,device):
        # tensor([[39., 20., 13., 39.],
        #         [39., 20., 2., 39.],
        #         [39., 20., 8., 39.],
        #         [39., 20., 20., 39.],
        #         [39., 20., 16., 39.]])   5,4

       # alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :,1]  # (1,1,39,2)  #直接在这加就可以了  torch.Size([1, 5, 39])

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
            #tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # print(tokens,inputs["input_ids"][0]) ['Bu', '.'] tensor([51827,3])
            inputs.to(device)
            outputs = self.LM(**inputs, labels=inputs["input_ids"])
            losses = outputs.losses  # [4.5008, 8.4680] Negative log likelihood   [(len-1)*bsz] Negative log likelihood.
            # prob = torch.exp(-losses)  #tensor([[0.0111, 0.0002]]   #prob= torch.prod(prob)  #2.3320e-06
            logp = -losses
            #print(logp)
            # prob= torch.prod(prob)
            # return prob.item()
            logp = logp.tolist()[0][-1]  # [[0.011100328527390957, 0.00021007754548918456]]
            result.append(logp)

        result_tensor = torch.tensor(result)
        return result_tensor  # 取最后一个的条概

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
