export CUDA_VISIBLE_DEVICES=4

python eval.py \
--eval_lrs3_model_file /home/xcpan/server_1/AAAIckp/av.ckpt \
--modal AV \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/L2_train/av_0.18.txt \
--beamWidth 5 \
--batch_size 64 \
--beta 0.18 \
--nbest 30 \


#nohup bash av_0.18.sh &