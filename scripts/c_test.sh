BATCH_SIZE=128 # This is also the default.

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --test True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_10000 --test_filename test_10000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_bpe_10000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --test True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_10000 --test_filename test_10000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_split_10000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --test True --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_10000 --test_filename test_10000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_merge_10000.log 

CUDA_VISIBLE_DEVICES=0 python code_nlm.py --test True --data_path data/py/ori --train_dir data/py/ori/model --test_filename test --gru True --batch_size 128 --word_level_perplexity True --cross_entropy True | tee py_logs/ent_ori_10000.log

