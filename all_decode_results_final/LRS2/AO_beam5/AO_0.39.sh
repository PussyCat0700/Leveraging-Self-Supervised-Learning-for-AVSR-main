export CUDA_VISIBLE_DEVICES=4

python eval.py \
--eval_lrs3_model_file /home/gryang/L2/newckpt/ao.ckpt \
--modal AO \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/L2_train/ao_0.39.txt \
--beamWidth 5 \
--batch_size 64 \
--beta 0.39 \
--nbest 30 \


#nohup bash AO_0.39.sh &