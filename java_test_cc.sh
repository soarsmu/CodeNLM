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

CUDA_VISIBLE_DEVICES=2 python code_nlm.py --test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True

CUDA_VISIBLE_DEVICES=2 python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE