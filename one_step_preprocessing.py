#Дополнительно загрузили ipywidgets 7.6.5

import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import tqdm

from utils import read_parquet_dataset_from_local
from dataset_preprocessing_utils import features, transform_credits_to_sequences, create_padded_buckets

pd.set_option("display.max_columns", None)

TRAIN_DATA_PATH = "data/train_data/"
TEST_DATA_PATH = "data/test_data/"

TRAIN_TARGET_PATH = "data/train_target.csv"
train_target = pd.read_csv(TRAIN_TARGET_PATH)

from collections import defaultdict

train_lens = []
test_lens = []
uniques = defaultdict(set)

for step in tqdm.notebook.tqdm(range(0, 4, 4),
                     desc="Count statistics on train data"):
        credits_frame = read_parquet_dataset_from_local(TRAIN_DATA_PATH, step, 4, verbose=True)
        seq_lens = credits_frame.groupby("id").agg(seq_len=("rn", "max"))["seq_len"].values
        train_lens.extend(seq_lens)
        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())
train_lens = np.hstack(train_lens)

for step in tqdm.notebook.tqdm(range(0, 2, 2),
                     desc="Count statistics on test data"):
        credits_frame = read_parquet_dataset_from_local(TEST_DATA_PATH, step, 2, verbose=True)
        seq_lens = credits_frame.groupby("id").agg(seq_len=("rn", "max"))["seq_len"].values
        test_lens.extend(seq_lens)
        credits_frame.drop(columns=["id", "rn"], inplace=True)
        for feat in credits_frame.columns.values:
            uniques[feat] = uniques[feat].union(credits_frame[feat].unique())
test_lens = np.hstack(test_lens)
uniques = dict(uniques)

keys_ = list(range(1, 59))
lens_ = list(range(1, 41)) + [45] * 5 + [50] * 5 + [58] * 8
bucket_info = dict(zip(keys_, lens_))
print(bucket_info)

for feat, uniq in uniques.items():
    print(f"Feature: {feat}, unique values: {uniq}")


def create_buckets_from_credits(path_to_dataset, bucket_info, save_to_path, frame_with_ids=None,
                                num_parts_to_preprocess_at_once: int = 1,
                                num_parts_total=50, has_target=False):
        block = 0
        for step in tqdm.notebook.tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                                       desc="Preparing credit data"):
                credits_frame = read_parquet_dataset_from_local(path_to_dataset, step, num_parts_to_preprocess_at_once,
                                                                verbose=True)
                credits_frame.loc[:, features] += 1
                seq = transform_credits_to_sequences(credits_frame)
                print("Transforming credits to sequences is done.")

                if frame_with_ids is not None:
                        seq = seq.merge(frame_with_ids, on="id")

                block_as_str = str(block)
                if len(block_as_str) == 1:
                        block_as_str = "00" + block_as_str
                else:
                        block_as_str = "0" + block_as_str

                processed_fragment = create_padded_buckets(seq, bucket_info=bucket_info, has_target=has_target,
                                                           save_to_file_path=os.path.join(save_to_path,
                                                                                          f"processed_chunk_{block_as_str}.pkl"))
                block += 1

train, val = train_test_split(train_target, random_state=42, test_size=0.1)


TRAIN_BUCKETS_PATH = "data/train_buckets_rnn"
VAL_BUCKETS_PATH = "data/val_buckets_rnn"
TEST_BUCKETS_PATH = "data/test_buckets_rnn"

#Не забыть почистить директории перед созданием файлов

create_buckets_from_credits(TRAIN_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=TRAIN_BUCKETS_PATH,
                            frame_with_ids=train,
                            num_parts_to_preprocess_at_once=4,
                            num_parts_total=4, has_target=True)



create_buckets_from_credits(TRAIN_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=VAL_BUCKETS_PATH,
                            frame_with_ids=val,
                            num_parts_to_preprocess_at_once=4,
                            num_parts_total=12, has_target=True)


create_buckets_from_credits(TEST_DATA_PATH,
                            bucket_info=bucket_info,
                            save_to_path=TEST_BUCKETS_PATH, num_parts_to_preprocess_at_once=2,
                            num_parts_total=2)



