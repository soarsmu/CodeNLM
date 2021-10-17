BATCH_SIZE=10 # This is also the default.

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --completion True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_2000 --test_filename test_2000 --identifier_map data/c-corpus/bigcode/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_bpe_2000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --completion True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_5000 --test_filename test_5000 --identifier_map data/c-corpus/bigcode/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_bpe_5000.log &

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_10000 --test_filename test_10000 --identifier_map data/c-corpus/bigcode/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_bpe_10000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --completion True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_2000 --test_filename test_2000 --identifier_map data/c-corpus/split/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_split_2000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --completion True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_5000 --test_filename test_5000 --identifier_map data/c-corpus/split/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_split_5000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --completion True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_10000 --test_filename test_10000 --identifier_map data/c-corpus/split/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_split_10000.log &

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_2000 --test_filename test_2000 --identifier_map data/c-corpus/merge/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_merge_2000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --completion True --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_5000 --test_filename test_5000 --identifier_map data/c-corpus/merge/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_merge_5000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --completion True --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_10000 --test_filename test_10000 --identifier_map data/c-corpus/merge/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_merge_10000.log &

