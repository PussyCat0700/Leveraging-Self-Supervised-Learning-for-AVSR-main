export CUDA_VISIBLE_DEVICES=7

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_0604-wer_0.054.ckpt \
--modal AO \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/beam40_AO/ao_0.88.txt \
--beamWidth 40 \
--batch_size 48 \
--beta 0.88 \
--nbest 30 \


#nohup bash beam40_AO/ao_0.88.sh &