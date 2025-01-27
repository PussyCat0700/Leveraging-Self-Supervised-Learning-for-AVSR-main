export CUDA_VISIBLE_DEVICES=7

python eval.py \
--decode_type HYBRID \
--type kenlm \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/bybrid_bs_1.txt \
--lmpath /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 1 \