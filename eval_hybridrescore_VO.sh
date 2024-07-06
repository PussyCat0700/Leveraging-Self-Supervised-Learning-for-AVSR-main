#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=rescore_VO
#SBATCH --partition=debug # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=eval_hybridrescore_VO.out
#SBATCH --error=eval_hybridrescore_VO.err

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_0604-wer_0.054.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /data1/yfliu/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/beam40_VO/ao_0.8.txt \
--beamWidth 40 \
--batch_size 48 \
--beta 0.8 \
--nbest 30