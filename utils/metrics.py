import editdistance
import numpy as np
import torch


def compute_error_ch(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):
    targetBatch = targetBatch.cpu()  #(361,)     #predictionBatch (85,) #predicitonLenBatch [15, 13,  7,  8, 13, 12,  8,  9]
    targetLenBatch = targetLenBatch.cpu() #tensor([50, 40, 25, 22, 40, 92, 75, 17])

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))  # 按每个句子长度分成 t(15,) t(13,) ...
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))  #同理
    totalEdits = 0
    totalChars = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()[:-1]
        trgt = trgts[n].numpy()[:-1]
        numEdits = editdistance.eval(pred, trgt)
        totalEdits = totalEdits + numEdits
        totalChars = totalChars + len(trgt)

    return totalEdits, totalChars


def compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):
    """
    Function to compute the Character Error Rate using the Predicted character indices and the Target character
    indices over a batch.
    CER is computed by dividing the total number of character edits (computed using the editdistance package)
    with the total number of characters (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the CER.
    """
    
    totalEdits, totalChars = compute_error_ch(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)

    return totalEdits / totalChars


def compute_error_word(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):  #当batchsize=12时  predictionBatch也是拼在一起的
    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))   #按各自的长度分成12个tensor，1个list
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
    totalEdits = 0
    totalWords = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()[:-1]    #去掉最后的39
        trgt = trgts[n].numpy()[:-1]    #去掉最后的39

        predWords = np.split(pred, np.where(pred == spaceIx)[0])  #按空格分成好几组 最终结果: 一个list 每遇到1另起一行  # np.where(pred == spaceIx)[0]:array([ 3, 14, 19, 23, 28])   # np.where(pred == spaceIx):(array([ 3, 14, 19, 23, 28]),)
        predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]  #numpy.ndarray.tostring: tobytes. Despite its name, it returns bytes not strs.

        trgtWords = np.split(trgt, np.where(trgt == spaceIx)[0])
        trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]

        numEdits = editdistance.eval(predWords, trgtWords)  #所有组
        totalEdits = totalEdits + numEdits
        totalWords = totalWords + len(trgtWords)  #6 一组一个词

    return totalEdits, totalWords


def compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):
    """
    Function to compute the Word Error Rate using the Predicted character indices and the Target character
    indices over a batch. The words are obtained by splitting the output at spaces.
    WER is computed by dividing the total number of word edits (computed using the editdistance package)
    with the total number of words (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
    """

    totalEdits, totalWords = compute_error_word(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx)

    return totalEdits / totalWords
