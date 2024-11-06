# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:42:43 2024

@author: casi
"""

import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython import get_ipython
from utils import read_parquet_dataset_from_local


ipython = get_ipython()
ipython.magic("%load_ext autoreload")
ipython.magic("%autoreload 2")


class CountAggregator(object):
    def __init__(self):
        self.encoded_feats = None

    def __extract_count_aggregations(
            self,
            dataframe: pd.DataFrame,
            mode: str) -> pd.DataFrame:
        feat_cols = list(dataframe.columns.values)
        feat_cols.remove("id")
        feat_cols.remove("rn")

        dummies = pd.get_dummies(dataframe[feat_cols], columns=feat_cols)
        dummy_feats = dummies.columns.values

        ohe_feats = pd.concat([dataframe, dummies], axis=1)
        ohe_feats = ohe_feats.drop(columns=feat_cols)

        ohe_feats.groupby("id")
        feats = ohe_feats.groupby(
            "id")[dummy_feats].sum().reset_index(drop=False)

        return feats

    def __transform_data(
            self,
            path_to_dataset: str,
            num_parts_to_preprocess_at_once: int = 1,
            num_parts_total: int = 50,
            mode: str = "fit_transform",
            save_to_path=None,
            verbose: bool = False):
        assert mode in [
            "fit_transform", "transform"], f"Unrecognized mode: {mode}. Available modes: fit_transform, transform"

        preprocessed_frames = []

        for step in tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                         desc="Transforming sequential data"):
            dataframe = read_parquet_dataset_from_local(
                path_to_dataset, start_from=step, num_parts_to_read=num_parts_to_preprocess_at_once, verbose=verbose)
            feats = self.__extract_count_aggregations(dataframe, mode)
            if save_to_path:
                feats.to_parquet(os.path.join(
                    save_to_path, f"processed_chunk_{step}.pq"))
                preprocessed_frames.append(feats)

        feats = pd.concat(preprocessed_frames)
        feats.fillna(np.uint8(0), inplace=True)
        dummies = list(feats.columns.values)
        dummies.remove("id")

        if (mode == "fit_transform"):
            self.encoded_feats = dummies
        else:
            assert not self.encoded_feats is None, "Transformer not fitted"
            for col in self.encoded_feats:
                if not col in dummies:
                    feats[col] = np.uint8(0)

        return feats[["id"]+self.encoded_feats]

    def fit_transform(
            self,
            path_to_dataset: str,
            num_parts_to_preprocess_at_once: int = 1,
            num_parts_total: int = 50,
            save_to_path=None,
            verbose: bool = False):
        return self.__transform_data(
            path_to_dataset=path_to_dataset,
            num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
            num_parts_total=num_parts_total,
            mode="fit_transform",
            save_to_path=save_to_path,
            verbose=verbose)

    def transform(
            self,
            path_to_dataset: str,
            num_parts_to_preprocess_at_once: int = 1,
            num_parts_total: int = 50,
            save_to_path=None,
            verbose: bool = False):
        return self.__transform_data(
            path_to_dataset=path_to_dataset,
            num_parts_to_preprocess_at_once=num_parts_to_preprocess_at_once,
            num_parts_total=num_parts_total,
            mode="transform",
            save_to_path=save_to_path,
            verbose=verbose)
