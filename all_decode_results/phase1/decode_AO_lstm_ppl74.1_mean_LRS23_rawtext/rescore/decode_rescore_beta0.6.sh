export CUDA_VISIBLE_DEVICES=7

python eval.py \
--decode_type HYBRID_RESCORE \
--type kenlm \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_rescore_beta0.6.txt \
--lmpath /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 48 \
--beta 0.6 \

#nohup bash decode_rescore_beta0.6.sh > decode_rescore_beta0.6.log &