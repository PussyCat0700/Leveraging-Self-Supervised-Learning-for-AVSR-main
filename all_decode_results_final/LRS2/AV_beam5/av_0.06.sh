export CUDA_VISIBLE_DEVICES=1

python eval.py \
--eval_lrs3_model_file /data2/alumni/xcpan/server_1/AAAIckp/av.ckpt \
--modal AV \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/L2_train/av_0.06.txt \
--beamWidth 5 \
--batch_size 64 \
--beta 0.06 \
--nbest 30 \


#nohup bash av_0.06.sh &