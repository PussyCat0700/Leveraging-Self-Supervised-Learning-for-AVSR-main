export CUDA_VISIBLE_DEVICES=7

python eval.py \
--eval_lrs3_model_file /home/gryang/L_store/check925/train-step_0108-wer_0.058.ckpt \
--modal AV \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/beam40_AV/av_0.4.txt \
--beamWidth 40 \
--batch_size 32 \
--beta 0.4 \
--nbest 30 \


#nohup bash beam40_AV/av_0.4.sh &