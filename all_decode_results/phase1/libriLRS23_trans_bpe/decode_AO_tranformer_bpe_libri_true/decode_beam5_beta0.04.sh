export CUDA_VISIBLE_DEVICES=5

python eval.py \
--decode_type HYBRID_LM \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_tranformer_bpe_libri_true/decode_beam5_beta0.04.txt \
--batch_size 48 \
--beta 0.04 \
--beamWidth 5 \
--nbest 30 \

# nohup bash decode_AO_tranformer_bpe_libri_true/decode_beam5_beta0.04.sh &