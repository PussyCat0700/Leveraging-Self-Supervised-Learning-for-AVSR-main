import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from data.lrs2_dataset import LRS2
from data.utils import collate_fn
from models.av_net import AVNet
from utils.general import inference

import hydra
from omegaconf import DictConfig

#这个好像跑不了 报错

def eval(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='wt.log', filemode='w')
    logger = logging.getLogger(__name__)
    
    logname=cfg.logname
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    # set seed
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    # check device
    torch.set_num_threads(args["NUM_CPU_CORE"])
    #torch.cuda.set_device(0)
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # declaring the test dataset and test dataloader
    #noiseParams = {"noiseFile": args["NOISE_FILE"], "noiseProb": 1 if args["TEST_WITH_NOISE"] else 0, "noiseSNR": args["TEST_NOISE_SNR_DB"]}
    noiseParams = {"noiseFile": args["HUMAN_NOISE_FILE"], "noiseProb": 1 if args["TEST_WITH_NOISE"] else 0,"noiseSNR": args["TEST_NOISE_SNR_DB"]}
    testData = LRS2(cfg.modal, "test", args["DATA_DIRECTORY"], args["HDF5_FILE"], args["CHAR_TO_INDEX"], args["STEP_SIZE"], False, noiseParams)
    testLoader = DataLoader(testData, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=False, **kwargs)



    if cfg.eval_lrs3_model_file is not None:    #重要！
        logger.info(cfg)
        
        #print("\nTrained Model File: %s" % (cfg.eval_lrs3_model_file))
        logger.info("\nTrained Model File: %s" % (cfg.eval_lrs3_model_file))

        if args["TEST_WITH_NOISE"]:
            #print("TEST_NOISE_SNR_DB: %s" % (args["TEST_NOISE_SNR_DB"]) )
            logger.info("TEST_NOISE_SNR_DB: %s" % (args["TEST_NOISE_SNR_DB"]))
        else:
            #print("no noise")
            logger.info("no noise")
            
        # declaring the model,loss function and loading the trained model weights
        modelargs = (args["DMODEL"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"], args["AUDIO_FEATURE_SIZE"],
                     args["VIDEO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["CHAR_NUM_CLASSES"])
        model = AVNet(cfg.modal, args['WAV2VEC_FILE'], args['MOCO_FRONTEND_FILE'], args["MAIN_REQ_INPUT_LENGTH"], modelargs)  #av_net.py
        stateDict = torch.load(cfg.eval_lrs3_model_file, map_location=device)['state_dict']
        msg = model.load_state_dict(stateDict, strict=False)
        #print(msg)
        logger.info(msg)
        model.to(device)

        #print("\nTesting the trained model .... \n")
        logger.info("\nTesting the trained model .... \n")

        inferenceParams = {"spaceIx": args["CHAR_TO_INDEX"][" "], "eosIx": args["CHAR_TO_INDEX"]["<EOS>"], "decodeType": cfg.decode_type,
                           "beamWidth": args["BEAM_WIDTH"], "modal": args["MODAL"], "Lambda": args["LAMBDA"]}

        testCER, testWER, testPER = inference(model, testLoader, device, logger, inferenceParams,cfg)
        #print("%sMODAL || Test CER: %.3f || Test WER: %.3f" % (args["MODAL"], testCER, testWER))
        logger.info("%sMODAL || Test CER: %.3f || Test WER: %.3f" % (args["MODAL"], testCER, testWER))

        #print("\nTesting Done.\n")
        logger.info("\nTesting Done.\n")

    else:
        print("Path to the trained model file not specified.\n")
        logger.info("Path to the trained model file not specified.\n")

    return

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    eval(cfg)

if __name__ == "__main__":
    main()
