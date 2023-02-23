import os

import torch
from tqdm import tqdm

from config import args
from .decoders import ctc_greedy_decode, teacher_forcing_attention_decode #flash_infer
from .metrics import compute_error_ch, compute_error_word


def index_to_string(indexBatch):
    return "".join([args["INDEX_TO_CHAR"][ix] if ix > 0 else "" for ix in indexBatch.tolist()])


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def inference(model, evalLoader, device, logger, inferenceParams,cfg):
    evalCER = 0
    evalWER = 0
    evalPER = 0
    evalCCount = 0
    evalWCount = 0
    evalPCount = 0

    Lambda = inferenceParams["Lambda"]  #args["LAMBDA"] 0.1
    if os.path.exists(args["CODE_DIRECTORY"] + "pred_%s.txt" % inferenceParams["decodeType"]):
        os.remove(args["CODE_DIRECTORY"] + "pred_%s.txt" % inferenceParams["decodeType"])
    if os.path.exists(args["CODE_DIRECTORY"] + "trgt.txt"):
        os.remove(args["CODE_DIRECTORY"] + "trgt.txt")

    model.eval()
    for batch, (inputBatch, targetinBatch, targetoutBatch, targetLenBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval", ncols=75)):
        if inferenceParams['modal'] == "AO":   #inputBatch ( (1,33792) audio, (1,33792) 全false mask , None, None )
            inputBatch = (inputBatch[0].float().to(device), inputBatch[1].to(device), None, None)
        elif inferenceParams['modal'] == "VO":
            inputBatch = (None, None, inputBatch[2].float().to(device), inputBatch[3].to(device))
        else:
            inputBatch = (inputBatch[0].float().to(device), inputBatch[1].to(device), inputBatch[2].float().to(device), inputBatch[3].to(device))
        targetinBatch = targetinBatch.int().to(device)  #(1,44)   ([39,.....])
        targetoutBatch = targetoutBatch.int().to(device)  #(1,44)   ([.....,39])
        targetLenBatch = targetLenBatch.int().to(device)  #[44]
        targetMask = torch.zeros((targetLenBatch.shape[0], targetLenBatch.max()), device=targetLenBatch.device) #(1,44) 全0
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1   #最后一项设为1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()  #全false？
        concatTargetoutBatch = targetoutBatch[~targetMask]

        with torch.no_grad():
            if inferenceParams["decodeType"] == "HYBRID":
                predictionBatch, predictionLenBatch = model.inference(inputBatch, False, device, Lambda, inferenceParams["beamWidth"], inferenceParams["eosIx"], 0)

            elif inferenceParams["decodeType"] == "HYBRID_RESCORE":   
                predictionBatch, predictionLenBatch = model.rescore(inputBatch, False, device, Lambda, inferenceParams["beamWidth"], inferenceParams["eosIx"], 0, inferenceParams["beta"],inferenceParams["nbest"],logger)                    
            
            elif inferenceParams["decodeType"] == "HYBRID_SHALLOW_FUSION":
                predictionBatch, predictionLenBatch = model.shallow_fusion(inputBatch, False, device, Lambda, inferenceParams["beamWidth"], inferenceParams["eosIx"], 0, inferenceParams["beta"])
                          
            elif inferenceParams["decodeType"] == 'HYBRID_ORACLE':  #可以batch_size>1
                nbest=100
                batch_size=cfg.batch_size
   
                predictionBatchs, predictionLenBatchs = model.oracle(inputBatch, False, device, Lambda, inferenceParams["beamWidth"], inferenceParams["eosIx"], 0)                
                
                concatTargetoutBatchs= concatTargetoutBatch
                targetLenBatchs= targetLenBatch
                TargetoutBatchslist =list(torch.split(concatTargetoutBatchs, targetLenBatchs.tolist()))   
                
                final_prediction=[]
                final_predictionlen=[]
                #for batch in range(batch_size):   IndexError: list index out of range 在最后一个batch报错 因为最后一个batch 不够48个！
                for batch in range(len(predictionBatchs)):
                    onesentence= predictionBatchs[batch]
                    sentencelen= predictionLenBatchs[batch]
                    onetarget = TargetoutBatchslist[batch]
                    targetlen = targetLenBatchs[batch]
                    
                    wer_list=[]
                    for i in range(nbest):
                        predictionBatch = onesentence[i] #tensor([20, 13,  3,  1,  2, 23,  2,  7,  3, 13,  5, 11, 11, 14,  1,  3,  9,  2,
                        predictionLenBatch = sentencelen[i]  #tensor(36, dtype=torch.int32)
                        w_edits, w_count = compute_error_word(predictionBatch, onetarget , predictionLenBatch, targetlen,inferenceParams["spaceIx"]) #0,6
                        
                        wer= float(w_edits) / w_count   
                        wer_list.append((wer,i))
                
                    sorted_wer_list=sorted(wer_list)
                    index=sorted_wer_list[0][1]
                    if index!=0:
                        logger.info("oracle index:"+str(index))
                    
                    final_prediction.append(onesentence[index])
                    final_predictionlen.append(sentencelen[index].item())
                    
                predictionBatch= torch.cat(final_prediction).int()
                predictionLenBatch = torch.tensor(final_predictionlen).int()
            
            # elif inferenceParams["decodeType"] == "FAIRSEQ_LM":
            #     inputLenBatch, outputBatch = model(inputBatch, targetinBatch, targetLenBatch.long(), False)    
            #     predictionBatch, predictionLenBatch,hyp_words  = flash_infer(outputBatch[0], inputLenBatch,inferenceParams["eosIx"],cfg,logger)   
                
            elif inferenceParams["decodeType"] == "ATTN":
                predictionBatch, predictionLenBatch = model.attentionAutoregression(inputBatch, False, device, inferenceParams["eosIx"])
            elif inferenceParams["decodeType"] == "TFATTN":
                inputLenBatch, outputBatch = model(inputBatch, targetinBatch, targetLenBatch.long(), False)
                predictionBatch, predictionLenBatch = teacher_forcing_attention_decode(outputBatch[1], inferenceParams["eosIx"])
            elif inferenceParams["decodeType"] == "CTC":
                inputLenBatch, outputBatch = model(inputBatch, targetinBatch, targetLenBatch.long(), False)
                predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0], inputLenBatch, inferenceParams["eosIx"])
            else:
                predictionBatch, predictionLenBatch = None, None

            predictionStr = index_to_string(predictionBatch).replace('<EOS>', '\n')            
            targetStr = index_to_string(concatTargetoutBatch).replace('<EOS>', '\n')
            
            #logger.info(predictionStr[:-1])
            #logger.info(targetStr[:-1])

            # with open("pred_%s_%s.txt" % (inferenceParams["decodeType"],str(cfg.batch_size)), "a") as f:
            #     f.write(predictionStr)

            # with open("trgt.txt", "a") as f:  
            #     f.write(targetStr)

            c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch)
            evalCER += c_edits
            evalCCount += c_count
            w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch, targetLenBatch,inferenceParams["spaceIx"])
            evalWER += w_edits
            evalWCount += w_count
            
            logger.info("\n")
            logger.info("evalWER:%d,evalWCount:%d" % (evalWER,evalWCount))
            print("batch%d || Test CER: %.5f || Test WER: %.5f" % (batch + 1, evalCER / evalCCount, evalWER / evalWCount))
            logger.info("batch%d || Test CER: %.5f || Test WER: %.5f" % (batch + 1, evalCER / evalCCount, evalWER / evalWCount))

    logger.info("evalWER:%d,evalCCount:%d" % (evalWER,evalWCount))
    evalCER /= evalCCount if evalCCount > 0 else 1
    evalWER /= evalWCount if evalWCount > 0 else 1
    evalPER /= evalPCount if evalPCount > 0 else 1
    return evalCER, evalWER, evalPER
