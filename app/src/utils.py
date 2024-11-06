# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:07:58 2024

@author: casi
"""

from typing import List
import pandas as pd
import os
from tqdm import tqdm
from fastparquet import ParquetFile


def read_parquet_dataset_from_local(
        path_to_dataset: str, start_from: int = 0,
        num_parts_to_read: int = 2, columns: List[str] = None,
        verbose: bool = False) -> pd.DataFrame:

    res = []
    start_from = max(0, start_from)
    dataset_paths = {
        int(os.path.splitext(filename)[0].split("_")[-1]):
            os.path.join(path_to_dataset, filename)
            for filename in os.listdir(path_to_dataset)
    }
    chunks = [dataset_paths[num] for num in sorted(
        dataset_paths.keys()) if num >= start_from][:num_parts_to_read]

    if verbose:
        print("Reading chunks:", *chunks, sep="\n")

    for chunk_path in tqdm(chunks, desc="Reading dataset with Pandas"):
        pf = ParquetFile(chunk_path)
        chunk = pf.to_pandas(columns)
        res.append(chunk)

    return pd.concat(res).reset_index(drop=True)
