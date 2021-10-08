#!/bin/bash
# Maximum training epochs
EPOCHS=50
# Initial learning rate
LR=0.1 # This is the default value. You can skip it if you don't want to change it.
# Training batch size
BATCH_SIZE=32 # This is also the default.
# RNN unroll timesteps for gradient calculation.
STEPS=200 # This also the default.
# 1 - Dropout probability
KEEP_PROB=0.5 # This is also the default.
# RNN hidden state size
STATE_DIMS=512 # This is also the default.
# Checkpoint and validation loss calculation frequency.
CHECKPOINT_EVERY=5000 # This is also the default.

# Train a small py model for 1 epoch.
CUDA_VISIBLE_DEVICES=0 python code_nlm.py --data_path data/python-corpus/split --train_dir data/python-corpus/split/model_2000 --train_filename small_training_set_pre_bpe_split_2000 --validation_filename validation_set_pre_bpe_split_2000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee py_logs/bpe_split_2000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --data_path data/python-corpus/split --train_dir data/python-corpus/split/model_5000 --train_filename small_training_set_pre_bpe_split_5000 --validation_filename validation_set_pre_bpe_split_5000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee py_logs/bpe_split_5000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --data_path data/python-corpus/split --train_dir data/python-corpus/split/model_10000 --train_filename small_training_set_pre_bpe_split_10000 --validation_filename validation_set_pre_bpe_split_10000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee py_logs/bpe_split_10000.log &

CUDA_VISIBLE_DEVICES=0 python code_nlm.py --data_path data/python-corpus/bigcode --train_dir data/python-corpus/bigcode/model_2000 --train_filename small_training_set_pre_enc_bpe_2000 --validation_filename validation_set_pre_enc_bpe_2000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee py_logs/bpe_2000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --data_path data/python-corpus/bigcode --train_dir data/python-corpus/bigcode/model_5000 --train_filename small_training_set_pre_enc_bpe_5000 --validation_filename validation_set_pre_enc_bpe_5000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee py_logs/bpe_5000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --data_path data/python-corpus/bigcode --train_dir data/python-corpus/bigcode/model_10000 --train_filename small_training_set_pre_enc_bpe_10000 --validation_filename validation_set_pre_enc_bpe_10000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee py_logs/bpe_10000.log &

# Testing the model (Calculating test set entropy) 
# CUDA_VISIBLE_DEVICES=2 python code_nlm.py --test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True

# # Code completion
# CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE

# # Dynamic Code Completion and Measuring Identifier Performance
# python code_nlm.py --completion True --dynamic True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE  --test_proj_filename $TEST_PROJ_NAMES_FILE --identifier_map $ID_MAP_FILE


