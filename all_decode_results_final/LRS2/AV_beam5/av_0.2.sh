export CUDA_VISIBLE_DEVICES=7

python eval.py \
--eval_lrs3_model_file /home/xcpan/server_1/AAAIckp/av.ckpt \
--modal AV \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/L2_train/av_0.2.txt \
--beamWidth 5 \
--batch_size 64 \
--beta 0.2 \
--nbest 30 \


#nohup bash av_0.2.sh &