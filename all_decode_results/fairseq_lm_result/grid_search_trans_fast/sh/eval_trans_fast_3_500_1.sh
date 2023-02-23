export CUDA_VISIBLE_DEVICES=1

python eval.py \
--type fairseqlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_trans_fast_3_500_1.txt \
--lmpath /mnt/lustre/sjtu/home/gry10/fairseq/language_model/LMcheckpoints/transformer_LRS3_second/checkpoint_best.pt \
--lexicon /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/pretrain_trainval.lst \
--beamthreshold 25.0 \
--wordscore 3 \
--beam 500 \
--lmweight 1 \


#nohup bash grid_search_trans_fast/sh/eval_trans_fast_3_500_1.sh > eval_trans_fast_3_500_1.log &