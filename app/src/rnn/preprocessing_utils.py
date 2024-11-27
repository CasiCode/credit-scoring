# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:03:57 2024

@author: casi
"""

from typing import Dict
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import FEATURES


def pad_sequence(array: np.ndarray, max_len: int) -> np.ndarray:
    if isinstance(max_len, float):
        print(max_len)
    res = np.zeros((len(FEATURES), max_len))
    res[:, :array.shape[1]] = array
    return res

def truncate(x, num_last_credits: int = 0):
    return pd.Series({"sequences": x.values.transpose()[:, -num_last_credits:]})

def transform_credits_to_seqs(frame: pd.DataFrame,
                               num_last_credits: int = 0) -> pd.DataFrame:
    return frame \
        .sort_values(["id", "rn"]) \
        .groupby(["id"])[FEATURES] \
        .apply(lambda x: truncate(x, num_last_credits = num_last_credits)) \
        .reset_index()

def create_padded_buckets(
        seqs_frame = pd.DataFrame, bucket_info: Dict[int, int],
        save_filepath: str = None, has_target: bool = True):
    seqs_frame["sequence_length"] = seqs_frame["sequences"].apply(lambda x: len(x[1]))
    seqs_frame["bucket_idx"] = seqs_frame["sequence_length"].map(bucket_info)
    padded_sequences = []
    targets = []
    ids = []
    
    for size, bucket in tqdm(seqs_frame.groupby("bucket_idx"), desc = "Extracting buckets"):
        padded_seqs = bucket["sequences"].apply(lambda x: pad_sequence(x, size)).values
        padded_sequences.append(np.stack(padded_seqs, axis = 0))
        
        if has_target: targets.append(bucket["flag"].values)
        
        ids.append(bucket["id"].values)
    
    seqs_frame.drop(columns=["bucket_idx"], inplace = True)
    
    res = {
        "id": np.array(ids, dtype=np.object)
        "padded_sequences": np.array(padded_seqs, dtype=np.object)
        "target": np.array(targets, dtype=np.object) if targets else []
    }
    
    if save_filepath:
        with open(save_filepath, "wb") as f:
            pickle.dump(res, f)
    
    return res