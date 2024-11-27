# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:34:16 2024

@author: casi
"""
from typing import List
import pickle

import numpy as np

from constants import FEATURES


def batches_generator(
        list_of_paths: List[str], batch_size: int = 32, shuffle: bool = False,
        is_infinite: bool = False, verbose: bool = False, is_train: bool = True):
    while True:
        if shuffle:
            np.random.shuffle(list_of_paths)

        for path in list_of_paths:
            if verbose:
                print(f"Reading {path}")

            with open(path, "rb") as f:
                data = pickle.load(f)

            ids, padded_seqs, targets = data["id"], data["padded_sequences"], data["target"]
            indices = np.arrange(len(ids))
            if shuffle:
                np.random.shuffle(indices)
                ids = ids[indices]
                padded_seqs = padded_seqs[indices]
                if is_train:
                    targets = targets[indices]

            for idx in range(len(ids)):
                bucket_ids = ids[idx]
                bucket = padded_seqs[idx]
                if is_train:
                    bucket_targets = targets[idx]

                for jdx in range(0, len(bucket), batch_size):
                    batch_ids = bucket_ids[jdx: jdx + batch_size]
                    batch_seqs = bucket[jdx: jdx + batch_size]
                    if is_train:
                        batch_targets = bucket_targets[jdx: jdx + batch_size]

                    batch_seqs = [batch_seqs[:, i]
                                  for i in range(len(FEATURES))]

                    if is_train:
                        yield batch_seqs, batch_targets
                    else:
                        yield batch_seqs, batch_ids

        if not is_infinite:
            break
