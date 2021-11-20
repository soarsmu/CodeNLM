# Code Structure

**scripts** contains all shell scripts for running experiments.

**reader.py** contains utility functions for reading data and providing batches for training and testing of models.

**code_nlm.py** contains the implementation of LMs for code and supports training, perplexity/cross-entropy calculation, code-completion simulation, measuring identifier specific performance for code completion.

# Usage

## Installation
```
# Python==3.6` is required, `Python>3.6` may not supported due to the tensorflow version.
pip install numpy==1.18.1 
pip install tensorflow-gpu==1.12.3
```

## Dataset and Processing

Training data: https://zenodo.org/record/3628638/files/small_training_set_pre?download=1
Validation: https://zenodo.org/record/3628638/files/validation_set_pre?download=1
Test: https://zenodo.org/record/3628638/files/c_test_set_pre?download=1

**data/split.py** use Ronin to split identifers

**data/merge.py** merge corpora.

The BPE implementation used can be found here: https://github.com/rsennrich/subword-nmt 

To apply byte pair encoding to word segmentation, invoke these commands:
```
subword-nmt learn-bpe -s {num_operations} < {train_file} > {codes_file}
subword-nmt apply-bpe -c {codes_file} < {test_file} > {out_file}
# num_operations = The number of BPE ops e.g., 10000
# train_file = The file on which to learn the encoding
# codes_file = The file in which to output the learned encoding
# test_file = The file to segment with the learned encoding
# out_file = The file in which to save the now segmented test_file
```

## Training

```
# Directory that contains train/validation/test data etc.
DATA_HOME=data/c/
# Directory in which the model will be saved.
MODEL_DIR==data/c/model
mkdir $MODEL_DIR

# Filenames
TRAIN_FILE=c_training_bpe_10000
VALIDATION_FILE=c_validation_bpe_10000
TEST_FILE=c_test_bpe_10000
ID_MAP_FILE=data/c/id_map_c_bpe_10000

# Maximum training epochs
EPOCHS=50
# Initial learning rate
LR=0.1
# Training batch size
BATCH_SIZE=32
# RNN unroll timesteps for gradient calculation.
STEPS=20 # 20-50 is a good range of values for dynamic experiments.
# 1 - Dropout probability
KEEP_PROB=0.5 
# RNN hidden state size
STATE_DIMS=512 
# Checkpoint and validation loss calculation frequency.
CHECKPOINT_EVERY=5000 

python code_nlm.py --data_path $DATA_HOME --train_dir $MODEL_DIR --train_filename $TRAIN_FILE --validation_filename $VALIDATION_FILE --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS
```

## Test Scenarios
### Test Entropy Calculation
```
# Calculating test set entropy
python code_nlm.py --test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True
```

### Test Code Completion
In this scenario the *batch_size* option is used to set the beam size.
```
python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE
```

Or identifiers only:

```
python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --identifier_map $ID_MAP_FILE --gru True --batch_size $BATCH_SIZE
```

