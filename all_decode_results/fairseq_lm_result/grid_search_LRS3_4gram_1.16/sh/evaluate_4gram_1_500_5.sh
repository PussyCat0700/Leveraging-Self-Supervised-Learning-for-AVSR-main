export CUDA_VISIBLE_DEVICES=2

python eval.py \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_4gram_1_500_5.txt \
--lmpath /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS3_4gram.bin \
--lexicon /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/pretrain_trainval.lst \
--beamthreshold 100.0 \
--wordscore 1 \
--beam 500 \
--lmweight 5 \
