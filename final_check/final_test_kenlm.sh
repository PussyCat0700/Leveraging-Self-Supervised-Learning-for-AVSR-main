export CUDA_VISIBLE_DEVICES=2

python eval.py \
--type kenlm \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_media/final_test_kenlm.txt \
--lmpath /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_media/LM_RELATED/bin/LRS23_4gram.bin \
--lexicon /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_media/LM_RELATED/lst/LRS23.lst \
--beamthreshold 100.0 \
--wordscore 2 \
--beam 500 \
--lmweight 2 \


# nohup bash grid_search1/sh/evaluate_4gram_2_500_2.sh > evaluate_4gram_2_500_2.log &