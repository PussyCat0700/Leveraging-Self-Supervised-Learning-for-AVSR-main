export CUDA_VISIBLE_DEVICES=3

python eval.py \
--type fairseqlm \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_media/final_test_fairseqlm.txt \
--lmpath /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_media/final_lm/checkpoint1130.pt \
--lexicon /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_media/LM_RELATED/lst/23plus.lst \
--beamthreshold 25.0 \
--wordscore 2 \
--beam 500 \
--lmweight 1 \


#nohup bash grid_search_trans_fast/sh/eval_trans_fast_1_500_1.sh  &