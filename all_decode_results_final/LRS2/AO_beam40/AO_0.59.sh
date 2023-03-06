export CUDA_VISIBLE_DEVICES=4

python eval.py \
--eval_lrs3_model_file /home/gryang/L2/newckpt/ao.ckpt \
--modal AO \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/L2_train/AO_beam40/AO_0.59.txt \
--beamWidth 40 \
--batch_size 64 \
--beta 0.59 \
--nbest 30 \


#nohup bash AO_beam40/AO_0.59.sh &