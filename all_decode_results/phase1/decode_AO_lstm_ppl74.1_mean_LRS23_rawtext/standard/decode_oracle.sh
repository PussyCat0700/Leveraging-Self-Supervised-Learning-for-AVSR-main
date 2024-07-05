export CUDA_VISIBLE_DEVICES=7

python eval.py \
--decode_type HYBRID_ORACLE \
--type kenlm \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/test.txt \
--lmpath /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \