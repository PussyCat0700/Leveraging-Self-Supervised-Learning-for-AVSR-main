export CUDA_VISIBLE_DEVICES=2

python eval.py \
--decode_type HYBRID_LM \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_result_new/decode_beam_beta0.05_beam10.txt \
--batch_size 48 \
--beta 0.05 \
--beamWidth 10\