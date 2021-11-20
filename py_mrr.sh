BATCH_SIZE=10 # This is also the default.

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path data/python-corpus/bigcode --train_dir data/python-corpus/bigcode/model_2000 --test_filename test_2000 --identifier_map data/python-corpus/bigcode/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_bpe_2000.log &

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path data/python-corpus/bigcode --train_dir data/python-corpus/bigcode/model_5000 --test_filename test_5000 --identifier_map data/python-corpus/bigcode/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_bpe_5000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --completion True --data_path data/python-corpus/bigcode --train_dir data/python-corpus/bigcode/model_10000 --test_filename test_10000 --identifier_map data/python-corpus/bigcode/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_bpe_10000.log &

CUDA_VISIBLE_DEVICES=2 python code_nlm.py --completion True --data_path data/python-corpus/split --train_dir data/python-corpus/split/model_2000 --test_filename test_2000 --identifier_map data/python-corpus/split/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_split_2000.log &

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path data/python-corpus/split --train_dir data/python-corpus/split/model_5000 --test_filename test_5000 --identifier_map data/python-corpus/split/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_split_5000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --completion True --data_path data/python-corpus/split --train_dir data/python-corpus/split/model_10000 --test_filename test_10000 --identifier_map data/python-corpus/split/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_split_10000.log &

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path data/python-corpus/merge --train_dir data/python-corpus/merge/model_2000 --test_filename test_2000 --identifier_map data/python-corpus/merge/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_merge_2000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --completion True --data_path data/python-corpus/merge --train_dir data/python-corpus/merge/model_5000 --test_filename test_5000 --identifier_map data/python-corpus/merge/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee py_logs/mrr_merge_5000.log &

CUDA_VISIBLE_DEVICES=0 python code_nlm.py --completion True --data_path split_model --train_dir split_model/model_10000 --test_filename test_split_10000 --identifier_map split_model/id_test_split_10000 --gru True --batch_size 10 | tee py_logs/mrr_split_10000.log &

