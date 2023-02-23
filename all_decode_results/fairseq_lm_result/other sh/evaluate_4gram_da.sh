export CUDA_VISIBLE_DEVICES=2

python eval.py \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_4gram.txt \
--lmpath /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS3_4gram.bin \
--lexicon /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/plus.lst \
--beam 1500 \
--beamthreshold 100.0 \