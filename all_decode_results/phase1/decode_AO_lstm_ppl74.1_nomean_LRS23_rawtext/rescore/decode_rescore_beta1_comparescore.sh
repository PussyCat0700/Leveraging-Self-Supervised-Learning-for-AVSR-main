export CUDA_VISIBLE_DEVICES=1

python eval.py \
--decode_type HYBRID_RESCORE \
--type kenlm \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneeds/decode_rescore_beta1_comparescore.txt \
--lmpath /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 48 \
--beta 1 \

#bash decode_rescore_beta1.sh


#这应该是我在改取不取mean的时候的check