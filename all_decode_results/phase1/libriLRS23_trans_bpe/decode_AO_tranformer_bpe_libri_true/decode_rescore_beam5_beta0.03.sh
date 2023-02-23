export CUDA_VISIBLE_DEVICES=4

python eval.py \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_tranformer_bpe_libri_true/decode_rescore_beam5_beta0.03.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.03 \
--nbest 30 \


#nohup bash decode_AO_tranformer_bpe_libri_true/decode_rescore_beam5_beta0.03.sh &