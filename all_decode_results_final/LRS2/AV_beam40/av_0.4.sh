export CUDA_VISIBLE_DEVICES=1

python eval.py \
--eval_lrs3_model_file /data2/alumni/xcpan/server_1/AAAIckp/av.ckpt  \
--modal AV \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/L2_train/AV_beam40/av_0.4.txt \
--beamWidth 40 \
--batch_size 64 \
--beta 0.4 \
--nbest 30 \


#nohup bash AV_beam40/av_0.4.sh &
