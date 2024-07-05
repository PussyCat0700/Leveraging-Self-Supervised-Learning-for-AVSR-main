export CUDA_VISIBLE_DEVICES=5

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/L2/newckpt/ao.ckpt \
--modal AO \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/L2_train/ao_0.txt \
--beamWidth 5 \
--batch_size 64 \
--beta 0 \
--nbest 30 \


#nohup bash AO_0.sh &