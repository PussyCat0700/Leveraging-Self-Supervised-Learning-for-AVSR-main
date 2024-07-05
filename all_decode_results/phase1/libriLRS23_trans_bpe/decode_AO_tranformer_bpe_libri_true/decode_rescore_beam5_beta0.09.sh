export CUDA_VISIBLE_DEVICES=1

python eval.py \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_tranformer_bpe_libri_true/decode_rescore_beam5_beta0.09.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.09 \
--nbest 30 \


#nohup bash decode_AO_tranformer_bpe_libri_true/decode_rescore_beam5_beta0.09.sh &