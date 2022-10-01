import os
import shutil
import sys  # 导入sys模块
import fairseq
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.plugins import *
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_warmup_lr import WarmupLR
from pytorch_lightning.strategies import DDPStrategy

from config import args
from data.lrs2_dataset import LRS2
from data.utils import collate_fn
from models.moco_visual_frontend import MoCoVisualFrontend
from models.utils import PositionalEncoding, conv1dLayers, outputConv, MaskedLayerNorm, generate_square_subsequent_mask
from utils.decoders import ctc_greedy_decode, teacher_forcing_attention_decode
from utils.label_smoothing import SmoothCTCLoss, SmoothCrossEntropyLoss
from utils.metrics import compute_error_ch, compute_error_word
from scheduler import  WarmupReduceLROnPlateau


class LRS2Lightning(pl.LightningDataModule):
    def __init__(self):
        super(LRS2Lightning, self).__init__()
        self.kwargs = {"num_workers": args["NUM_WORKERS"], "persistent_workers": True if args["NUM_WORKERS"] > 0 else False, "pin_memory": True}

    def setup(self, stage):
        if stage == "fit" or stage is None:
            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": args["NOISE_PROBABILITY"], "noiseSNR": args["NOISE_SNR_DB"]}   #'/home/xcpan/LRS2/mvlrs_v1/Noise.h5'  0.25 5
            self.trainData = LRS2(args['MODAL'], "train", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                                  True, noiseParams)

            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 0, "noiseSNR": args["NOISE_SNR_DB"]}
            self.valData = LRS2(args['MODAL'], "val", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False,
                                noiseParams)

        if stage == "test" or stage is None:
            noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 0, "noiseSNR": args["NOISE_SNR_DB"]}
            self.testData = LRS2(args['MODAL'], "test", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False,
                                 noiseParams)

    def train_dataloader(self):
        return DataLoader(self.trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **self.kwargs)


class AVNet(pl.LightningModule):

    def __init__(self, modal, W2Vfile, MoCofile, reqInpLen, modelargs):
        super(AVNet, self).__init__()   #' W2Vfile: /home/xcpan/pretrain_model/wav2vec_vox_new.pt'  'MoCofile: /home/xcpan/pretrain_model/moco_frontend.pt'

        self.trainParams = {"spaceIx": args["CHAR_TO_INDEX"][" "], "eosIx": args["CHAR_TO_INDEX"]["<EOS>"], "modal": args["MODAL"],
                            "Alpha": args["ALPHA"]}   #{'spaceIx': 1, 'eosIx': 39, 'modal': 'AV', 'Alpha': 0.2}

        self.valParams = {"spaceIx": args["CHAR_TO_INDEX"][" "], "eosIx": args["CHAR_TO_INDEX"]["<EOS>"], "modal": args["MODAL"],
                          "Alpha": args["ALPHA"]}  #{'spaceIx': 1, 'eosIx': 39, 'modal': 'AV', 'Alpha': 0.2}

        self.ft = False

        self.CTCLossFunction = [SmoothCTCLoss(args["CHAR_NUM_CLASSES"], blank=0)]
        self.CELossFunction = [SmoothCrossEntropyLoss()]

        dModel, nHeads, numLayers, peMaxLen, audinSize, vidinSize, fcHiddenSize, dropout, numClasses = modelargs
        self.save_hyperparameters()   #新加
        self.modal = modal
        self.numClasses = numClasses  #40
        self.reqInpLen = reqInpLen  #80
        # A & V Modal
        tx_norm = nn.LayerNorm(dModel)  #transformer的v
        self.maskedLayerNorm = MaskedLayerNorm()    #layer_norm = nn.LayerNorm(embedding_dim)  layer_norm(embedding)
        if self.modal == "AV":
            self.ModalityNormalization = nn.LayerNorm(dModel)
        self.EncoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)  #512,500
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
            wav2vecModel = wav2vecModel[0]   # 本身的模型巨大 # 一堆参数 cfg: Namespace(activation_dropout=0.1, activation_fn='gelu', adam_betas='(0.9,0.98)', adam_eps=1e-06, apply_mask=True, arch='wav2vec2',
            wav2vecModel.remove_pretraining_modules()    #task： <fairseq.tasks.audio_pretraining.AudioPretrainingTask object at 0x7f2eeadf5100>
            self.wav2vecModel = wav2vecModel
            # back-end
            self.audioConv = conv1dLayers(self.maskedLayerNorm, audinSize, dModel, dModel, downsample=True)    #dropout=0.1
            audioEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout) #这是一层 #dim_feedforward – the dimension of the feedforward network model
            self.audioEncoder = nn.TransformerEncoder(audioEncoderLayer, num_layers=numLayers, norm=tx_norm)
        else:
            self.wav2vecModel = None
            self.audioConv = None
            self.audioEncoder = None
        # visual
        if not self.modal == "AO":
            # front-end
            visualModel = MoCoVisualFrontend()
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
            jointEncoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)   #高度怀疑这三个transformer 结构一样
            self.jointEncoder = nn.TransformerEncoder(jointEncoderLayer, num_layers=numLayers, norm=tx_norm)
        self.jointOutputConv = outputConv(self.maskedLayerNorm, dModel, numClasses)   #这个是下面那个decoder  最后output dim是 numClasses40 由conv 非线性 norm组成
        self.decoderPositionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)  # transformer decoder
        self.embed = torch.nn.Sequential(   #transformer 结构图不是 output embeding + positional encoding 嘛
            nn.Embedding(numClasses, dModel),   #40,512
            self.decoderPositionalEncoding
        )
        jointDecoderLayer = nn.TransformerDecoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.jointAttentionDecoder = nn.TransformerDecoder(jointDecoderLayer, num_layers=numLayers, norm=tx_norm)
        self.jointAttentionOutputConv = outputConv("LN", dModel, numClasses)
        return

    def subNetForward(self, inputBatch, maskw2v):
        audioBatch, audMask, videoBatch, vidLen = inputBatch  #(8,97280) (8,97280) (8,152,1,112,112) (8,)
        if not self.modal == "VO":   #(8，2  41664） (8，241664） 这是batch1 没报错 报错的太tm长了 （8，398336）
            if self.ft and self.modal == "AO":
                audioBatch, audMask = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v)
            else:
                with torch.no_grad():   #（8，754，1024） （8，754）  #wav2vec 每20ms产生一个向量表征
                    audioBatch, audMask = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v)  #(8,303,1024)  #(8,303)

            audLen = torch.sum(~audMask, dim=1) #(8,) tensor([299, 754, 146, 106,  81, 310, 119,  68] 最大值是754！ 长度和是1883
        else:
            audLen = None

        if not self.modal == "AO":
            videoBatch = videoBatch.transpose(1, 2)
            if self.modal == "AV":
                with torch.no_grad():
                    videoBatch = self.visualModel(videoBatch, vidLen.long())  #(927,2048)
            else:
                videoBatch = self.visualModel(videoBatch, vidLen.long())
                                # vidLen tensor([ 35, 151,  94, 152, 146,  59, 139, 151], device='cuda:0',
            videoBatch = list(torch.split(videoBatch, vidLen.tolist(), dim=0))  #list:8   分别是 (35,2048) (151,2048) ...

        audioBatch, videoBatch, inputLenBatch, mask = self.makePadding(audioBatch, audLen, videoBatch, vidLen)
                                                    #(8,304,1024)           #tensor([ 80, 152,  95, 152, 147,  80, 139, 152]  #mask:(8,152)
        if isinstance(self.maskedLayerNorm, MaskedLayerNorm):
            self.maskedLayerNorm.SetMaskandLength(mask, inputLenBatch)

        if not self.modal == "VO":
            if self.modal == "AV":
                with torch.no_grad():   #(8,754,1024)
                    audioBatch = audioBatch.transpose(1, 2)  #(8,1024,304)
                    audioBatch = self.audioConv(audioBatch)   #（8, 512,512)   #到backend的conv1d了
                    audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)   #(152,8,512)
                    audioBatch = self.EncoderPositionalEncoding(audioBatch)  #(152,8,512)
                    audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask) #(152,8,512)
            else:
                audioBatch = audioBatch.transpose(1, 2)  #(8,1024,754)  #我觉得这个754 就是采样的样本数  （4，1024，304）  看最右边这一组  其实要看304
                audioBatch = self.audioConv(audioBatch)  #(8,512,152)  也是这个                             #(4,512,152)    #152 就是那个长度
                audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)  #(377,8,512)  #377是754减一半，频率对齐     （152，4，512）
                #print("Audio长度：", audioBatch.shape[0])
                audioBatch = self.EncoderPositionalEncoding(audioBatch)   #(377,8,512)
                audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask)  #(377,8,512)

        if not self.modal == "AO":
            if self.modal == "AV":
                with torch.no_grad():  #(8,152,2048)   #2048是特征dim
                    videoBatch = videoBatch.transpose(1, 2)  #(8,2048,152)
                    videoBatch = self.videoConv(videoBatch)  # (8,512,152)
                    videoBatch = videoBatch.transpose(1, 2).transpose(0, 1)  #(152,8,512)
                    videoBatch = self.EncoderPositionalEncoding(videoBatch)  #(152,8,512)
                    videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask) #(152,8,512)
            else:
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
            jointBatch = torch.cat([self.ModalityNormalization(audioBatch), self.ModalityNormalization(videoBatch)], dim=2)  #(152,8,1024)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)   #(8,1024,152)
            jointBatch = self.jointConv(jointBatch)  #(8,512,152)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)  #(152,8,512)
            jointBatch = self.EncoderPositionalEncoding(jointBatch)  #(152,8,512)
            jointBatch = self.jointEncoder(jointBatch, src_key_padding_mask=mask) #(152,8,512)

        return jointBatch, inputLenBatch, mask

    def forward(self, inputBatch, targetinBatch, targetLenBatch, maskw2v):  # tuple:4 ; (8,92) ; (8,) ; false
        jointBatch, inputLenBatch, mask = self.subNetForward(inputBatch, maskw2v)  #(152,8,512) tuple:4 (8,512)
        jointCTCOutputBatch = jointBatch.transpose(0, 1).transpose(1, 2) #(8,512,512)  #AO: (8,512,152)
        jointCTCOutputBatch = self.jointOutputConv(jointCTCOutputBatch)  #(8,40,152)  #AO: (8,40,152)
        jointCTCOutputBatch = jointCTCOutputBatch.transpose(1, 2).transpose(0, 1)  #(152,8,40)  #AO: (152,8,40)
        jointCTCOutputBatch = F.log_softmax(jointCTCOutputBatch, dim=2) #(152,8,40) #AO: (152,8,40)  ???152

        targetinBatch = self.embed(targetinBatch.transpose(0, 1)) # targetinBatch 原本是 (8,92)   #（92, 8, 512）
        targetinMask = self.makeMaskfromLength(targetinBatch.shape[:-1][::-1], targetLenBatch, self.device)  #(8,92)
        squareMask = generate_square_subsequent_mask(targetinBatch.shape[0], self.device)  #(92,92)
        jointAttentionOutputBatch = self.jointAttentionDecoder(targetinBatch, jointBatch, tgt_mask=squareMask,    #nn.TransformerDecoder
                                                               tgt_key_padding_mask=targetinMask, memory_key_padding_mask=mask) #( 92, 8, 512）
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(0, 1).transpose(1, 2)    #(8,512,92)
        jointAttentionOutputBatch = self.jointAttentionOutputConv(jointAttentionOutputBatch)     #(8,40,92)
        jointAttentionOutputBatch = jointAttentionOutputBatch.transpose(1, 2)  #(8,92,40)

        outputBatch = (jointCTCOutputBatch, jointAttentionOutputBatch)  #(152,8,40) (8,92,40)
        return inputLenBatch, outputBatch

    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask

    def makePadding(self, audioBatch, audLen, videoBatch, vidLen):
        if self.modal == "AO":
            audPadding = audLen % 2
            mask = (audPadding + audLen) > 2 * self.reqInpLen #tensor([False,  True,  True,  True,  True, False,  True,  True],
            audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)   #tensor([89,  1,  1,  1,  0, 41,  0,  1], device='cuda:3')
            audLeftPadding = torch.floor(torch.div(audPadding, 2)).int()  #tensor([44,  0,  0,  0,  0, 20,  0,  0],
            audRightPadding = torch.ceil(torch.div(audPadding, 2)).int() #tensor([45,  1,  1,  1,  0, 21,  0,  1],

            audioBatch = audioBatch.unsqueeze(1).unsqueeze(1)  #(8,1,1,303,1024)
            audioBatch = list(audioBatch)  #搞成8个list
            for i, _ in enumerate(audioBatch):   #这里！！！！！！！！1
                pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding[i], audRightPadding[i]))
                audioBatch[i] = pad(audioBatch[i][:, :, :audLen[i]]).squeeze(0).squeeze(0)    #这会报错  RuntimeError: Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: [1, 1, 0, 1024]

            audioBatch = pad_sequence(audioBatch, batch_first=True)  #（8,304,1024）
            inputLenBatch = ((audLen + audPadding) // 2).long() #tensor([ 80, 152,  95, 152, 147,  80, 139, 152] )
            mask = self.makeMaskfromLength([audioBatch.shape[0]] + [audioBatch.shape[1] // 2], inputLenBatch, self.device)

        elif self.modal == "VO":
            vidPadding = torch.zeros(len(videoBatch)).long().to(self.device)

            mask = (vidPadding + vidLen) > self.reqInpLen
            vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen)

            vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
            vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()

            for i, _ in enumerate(videoBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
                videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            videoBatch = pad_sequence(videoBatch, batch_first=True)
            inputLenBatch = (vidLen + vidPadding).long()
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, self.device)

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
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, self.device)

        return audioBatch, videoBatch, inputLenBatch, mask

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))            #scheduler = WarmupLR(scheduler_reduce, init_lr=args["FINAL_LR"], num_warmup=args["LRS2_WARMUP_PERIOD"], warmup_strategy='cos')
        scheduler = WarmupReduceLROnPlateau(optimizer)
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance (required)
                'interval': 'epoch',  # The unit of the scheduler's step size
                'frequency': 1,  # The frequency of the scheduler
                'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
                'monitor': 'info/val_WER',
                'strict': True,  # Whether to crash the training if `monitor` is not found
                'name': None,  # Custom name for LearningRateMonitor to use
            }
        }
        return optim_dict

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
    #     # scheduler_reduce = ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"], patience=args["LR_SCHEDULER_WAIT"],  #Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.
    #     #                                      threshold=args["LR_SCHEDULER_THRESH"], threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
    #     #if args["LRW_WARMUP_PERIOD"] > 0:   #80
    #         #scheduler = WarmupLR(scheduler_reduce, init_lr=args["FINAL_LR"], num_warmup=args["LRS2_WARMUP_PERIOD"], warmup_strategy='cos')
    #     scheduler = WarmupReduceLROnPlateau(optimizer)  #说可能是版本依赖问题
    #         #scheduler.step(1)   #因为一下改了learning rate 就出错了  我觉得这块还是有问题
    #     # else:
    #     #     scheduler = scheduler_reduce
    #
    #     optim_dict = {
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': scheduler,  # The LR scheduler instance (required)
    #             'interval': 'epoch',  # The unit of the scheduler's step size
    #             'frequency': 1,  # The frequency of the scheduler
    #             'reduce_on_plateau': True,  # For ReduceLROnPlateau scheduler
    #             'monitor': 'info/val_WER',
    #             'strict': True,  # Whether to crash the training if `monitor` is not found
    #             'name': None,  # Custom name for LearningRateMonitor to use
    #         }
    #     }
    #     # default_lrs = list()
    #     # for param_group in optimizer.param_groups:
    #     #     default_lrs.append(param_group['lr'])
    #     # print(default_lrs)
    #     #[0.0001]  init_lr=1e-4
    #
    #     return optim_dict

    def training_step(self, batch, batch_idx):
        inputBatch, targetinBatch, targetoutBatch, targetLenBatch = batch  #{list:4} (8,216064) (8,216064) (none,none)
        Alpha = self.trainParams["Alpha"]   #(8,31)  #(8,31)  #tensor(8,) [17, 26, 16, 22, 18, 31, 29, 24]

        if self.trainParams['modal'] == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.trainParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())
        targetinBatch = targetinBatch.int()
        targetoutBatch = targetoutBatch.int()
        targetLenBatch = targetLenBatch.int()
        targetMask = torch.zeros_like(targetoutBatch, device=self.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()
        concatTargetoutBatch = targetoutBatch[~targetMask]  #(183,)

        inputLenBatch, outputBatch = self(inputBatch, targetinBatch, targetLenBatch.long(), True)
        with torch.backends.cudnn.flags(enabled=False):
            ctcloss = self.CTCLossFunction[0](outputBatch[0], concatTargetoutBatch, inputLenBatch, targetLenBatch)
            celoss = self.CELossFunction[0](outputBatch[1], targetoutBatch.long())
            loss = Alpha * ctcloss + (1 - Alpha) * celoss
        self.log("info/train_ctcloss", ctcloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,batch_size=args["BATCH_SIZE"])   #这块tensorboard就可以搞了
        self.log("info/train_celoss", celoss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,batch_size=args["BATCH_SIZE"])
        self.log("info/train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,batch_size=args["BATCH_SIZE"])  #prog_bar: Logs to the progress bar (Default: False).

        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0].detach(), inputLenBatch, self.trainParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/train_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,batch_size=args["BATCH_SIZE"])
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.trainParams["spaceIx"])
        self.log("info/train_WER", w_edits / w_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True,batch_size=args["BATCH_SIZE"])
        return loss
        #return {'loss': loss,'celoss':celoss}

    def validation_step(self, batch, batch_idx):
        inputBatch, targetinBatch, targetoutBatch, targetLenBatch = batch  # batch: {list:4} 0:{list:4} 1 (8,92) 2(8,92) 3 (8,)
        Alpha = self.trainParams["Alpha"]  #0.2     # [torch.Size([8, 97280]), torch.Size([8, 97280]),torch.Size([8, 152, 1, 112, 112]),torch.Size([8])

        if self.valParams['modal'] == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.valParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())
        targetinBatch = targetinBatch.int() #(8,92)
        targetoutBatch = targetoutBatch.int() #(8,92)
        targetLenBatch = targetLenBatch.int() #8
        targetMask = torch.zeros_like(targetoutBatch, device=self.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()   # (8,92)n true false
        concatTargetoutBatch = targetoutBatch[~targetMask]  #(361,)
        #outputBatch 是一个tuple 0 （152,8,40) 1 （8，92，40）
        inputLenBatch, outputBatch = self(inputBatch, targetinBatch, targetLenBatch.long(), False)  #inputlenBatch tensor([ 80, 152,  95, 152, 147,  80, 139, 152], device='cuda:3') # inputLenBatch: tensor([ 80, 152,  95, 152, 147,  80, 139, 152], device='cuda:0')

        with torch.backends.cudnn.flags(enabled=False):    # outputbatch tuple 0 (152,8,40) (8,92,40)
            ctcloss = self.CTCLossFunction[0](outputBatch[0], concatTargetoutBatch, inputLenBatch, targetLenBatch) #tensor(11.3492, device='cuda:0')
            celoss = self.CELossFunction[0](outputBatch[1], targetoutBatch.long())
            loss = Alpha * ctcloss + (1 - Alpha) * celoss  # tensor(5.3786, device='cuda:0') #tensor(3.8859, device='cuda:0')
        self.log("info/val_ctcloss", ctcloss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True ,batch_size=args["BATCH_SIZE"])
        self.log("info/val_celoss", celoss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True ,batch_size=args["BATCH_SIZE"])
        self.log("info/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True ,batch_size=args["BATCH_SIZE"])

        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0], inputLenBatch, self.valParams["eosIx"])  #tensor(85,)  (8,) [15,13,7,8,13,12,8,9] #和是85
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/val_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True ,batch_size=args["BATCH_SIZE"])
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.valParams["spaceIx"])
        self.log("info/val_WER", w_edits / w_count, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True ,batch_size=args["BATCH_SIZE"])

        predictionBatch, predictionLenBatch = teacher_forcing_attention_decode(outputBatch[1], self.valParams["eosIx"])
        c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
        self.log("CER/val_TF_CER", c_edits / c_count, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True ,batch_size=args["BATCH_SIZE"])
        w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch, self.valParams["spaceIx"])
        self.log("info/val_TF_WER", w_edits / w_count, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True ,batch_size=args["BATCH_SIZE"])

        #return {'info/val_loss': loss}
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        #print(items)
        items.pop("v_num", None)  # items里就只剩 {'loss': '0.534', 'v_num': 34}->{'loss': '0.534'}
        #print(items)
        #exit(0)   #测一下这个
        return items


class UnfreezeCallback(Callback):
    """Unfreeze feature extractor callback."""

    def on_epoch_start(self, trainer, pl_module):
        if not pl_module.ft and trainer.current_epoch > args["W2V_FREEZE_EPOCH"]:  #40
            pl_module.ft = True

    def on_train_epoch_start(self, trainer, pl_module):
        if not pl_module.ft:
            pl_module.wav2vecModel.eval()
        if args["MODAL"] == "AV":
            pl_module.wav2vecModel.eval()
            pl_module.audioConv.eval()
            pl_module.audioEncoder.eval()
            pl_module.visualModel.eval()
            pl_module.videoConv.eval()
            pl_module.videoEncoder.eval()


def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pl.seed_everything(args["SEED"])
    torch.set_num_threads(args["NUM_CPU_CORE"])
    LRS2Dataloader = LRS2Lightning()
    LRS2Dataloader.setup('fit')
    modelargs = (args["DMODEL"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"], args["AUDIO_FEATURE_SIZE"],
                 args["VIDEO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["CHAR_NUM_CLASSES"])   #(512, 8, 6, 500, 1024, 2048, 2048, 0.1, 40)
    model = AVNet(args['MODAL'], args['WAV2VEC_FILE'], args['MOCO_FRONTEND_FILE'], args["MAIN_REQ_INPUT_LENGTH"], modelargs)

    # model = AVNet.load_from_checkpoint(checkpoint_path="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/train-step_0078-wer_0.142.ckpt",
                                     # modal=args['MODAL'],W2Vfile=args['WAV2VEC_FILE'], MoCofile= args['MOCO_FRONTEND_FILE'],reqInpLen=args["MAIN_REQ_INPUT_LENGTH"], modelargs=modelargs )
    #model = AVNet.load_from_checkpoint(checkpoint_path="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/train-step_0001-wer_1.000.ckpt")


    #checkpoint = torch.load("/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/train-step_0079-wer_0.109.ckpt")
    #model.load_state_dict(checkpoint['model_state_dict'])

    # loading the pretrained weights
    if not args["MODAL"] == "AV" and args["TRAIN_LRS2_MODEL_FILE"] is not None:
        stateDict = torch.load(args["TRAIN_LRS2_MODEL_FILE"], map_location="cpu")['state_dict']  #When you call torch.load() on a file which contains GPU tensors, those tensors will be loaded to GPU by default. You can call torch.load(.., map_location='cpu') and then load_state_dict() to avoid GPU RAM surge when loading a model checkpoint.
        model.load_state_dict(stateDict, strict=False)

    if args["MODAL"] == "AV" and args["TRAINED_AO_FILE"] is not None and args["TRAINED_VO_FILE"] is not None:
        AOstateDict = torch.load(args["TRAINED_AO_FILE"])['state_dict']
        stateDict = torch.load(args["TRAINED_VO_FILE"])['state_dict']
        for k in list(AOstateDict.keys()):
            if not (k.startswith('audioConv') or k.startswith('wav2vecModel')):
                del AOstateDict[k]

        for k in list(stateDict.keys()):
            if not (k.startswith('videoConv') or k.startswith('visualModel')):
                del stateDict[k]
        stateDict.update(AOstateDict)
        model.load_state_dict(stateDict, strict=False)

    writer = pl_loggers.TensorBoardLogger(save_dir=args["CODE_DIRECTORY"], name='log', default_hp_metric=False)
    # removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + "checkpoints"):
        shutil.rmtree(args["CODE_DIRECTORY"] + "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args["CODE_DIRECTORY"] + "checkpoints/models",   #好像每个epoch都会存一个
        filename=
        "train-step_{epoch:04d}-cer_{CER/val_CER:.3f}" if args["LR_SCHEDULER_METRICS"] == "CER" else "train-step_{epoch:04d}-wer_{info/val_WER:.3f}",
        monitor='CER/val_CER' if args["LR_SCHEDULER_METRICS"] == "CER" else 'info/val_WER',   #config 里是WER
        every_n_epochs=1,
        every_n_train_steps=0,
        save_top_k=10,   #记录best k个模型
        mode="min",
        auto_insert_metric_name=False,
        save_weights_only=False   #等于false的话 会报奇奇怪怪的问题
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args["MODAL"] == "VO":
        callback_list = [checkpoint_callback, lr_monitor]
    else:
        callback_list = [checkpoint_callback, lr_monitor, UnfreezeCallback()]

    trainer = pl.Trainer(
        gpus=args["GPU_ID"],    #max_epochs=50来设置 默认是1000
        max_epochs=800,
        benchmark=False,
        deterministic=False,
        logger=writer,
        default_root_dir=args["CODE_DIRECTORY"],
        callbacks=callback_list,
        #accelerator="ddp",
        #plugins=DDPPlugin(find_unused_parameters=False if args["MODAL"] == "VO" else True),
        strategy=DDPStrategy(find_unused_parameters=False if args["MODAL"] == "VO" else True)

    )


    #trainer.fit(model, LRS2Dataloader)
    trainer.fit(model, LRS2Dataloader,ckpt_path="/home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/train-step_0020-wer_0.992.ckpt")
    return


if __name__ == "__main__":
    main()
