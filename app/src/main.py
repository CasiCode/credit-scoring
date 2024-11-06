# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:02:34 2024

@author: casi
"""

import os
import sys
from IPython import get_ipython

import numpy as np
import pandas as pd
from CountAggregator import CountAggregator

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from tqdm import tqdm
import constants

pd.set_option('display.max_columns', None)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main():
    pd.set_option('display.max_columns', None)

    sys.path.append("../")

    """
    dataframe = read_parquet_dataset_from_local(
        constants.TRAIN_DATA_PATH, start_from=0, num_parts_to_read=1)

    memory_usage = dataframe.memory_usage(index=True).sum() / 10**9
    expected_memory_usage = memory_usage * 12
    print(f"Memory usage for one partation: {round(memory_usage, 3)} GB")
    print(
        f"Expected memory usage for the whole dataset: {round(expected_memory_usage, 3)} GB")

    del dataframe
    gc.collect()
    """

    ipython = get_ipython()
    ipython.magic("%%time")

    aggregator = CountAggregator()
    train_data = aggregator.fit_transform(
        constants.TRAIN_DATA_PATH,
        num_parts_total=12,
        save_to_path=constants.TRAIN_FEATURES_PATH,
        verbose=True)

    ram_usage = train_data.memory_usage(index=True).sum() / 1e9
    print(f"Missing values count: {train_data.isna().sum().sum()}")
    print(
        f"RAM usage for the dataframe with features: {round(ram_usage, 3)} GB")

    test_data = aggregator.transform(
        constants.TEST_DATA_PATH,
        num_parts_to_preprocess_at_once=2,
        num_parts_total=2,
        save_to_path=constants.TEST_FEATURES_PATH,
        verbose=True)

    train_target = pd.read_csv(constants.TRAIN_TARGET_PATH)
    train_data_target = train_target.merge(train_data, on="id")

    feat_cols = list(train_data_target.columns.values)
    feat_cols.remove("id")
    feat_cols.remove("flag")

    targets = train_data_target["flag"].values

    cv = KFold(n_splits=5, random_state=100, shuffle=True)
    oof = np.zeros(len(train_data_target))
    train_preds = np.zeros(len(train_data_target))

    models = []

    tree_parameters = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 5,
        "reg_lambda": 1,
        "num_leaves": 64,
        "n_jobs": 5,
        "n_estimators": 2000
    }

    for fold_, (train_idx, val_idx) in enumerate(cv.split(train_data_target, targets), 1):
        print(f"Training with fold {fold_} started")
        lgb_model = lgb.LGMBClassifier(**tree_parameters)
        train, val = train_data_target.iloc[train_idx], train_data_target.iloc[val_idx]

        lgb_model.fit(train[feat_cols], train.flag.values, eval_set=[(val[feat_cols], val.flag.values)],
                      early_stopping_rounds=50, verbose=50)

        oof[val_idx] = lgb_model.predict_proba(val[feat_cols])[:, 1]
        train_preds[train_idx] += lgb_model.predict_proba(train[feat_cols])[
            :, 1]/(cv.n_splits-1)
        models.append(lgb_model)
        print(f"Training with fold {fold_} completed")

    print("Train ROC-AUC: ", roc_auc_score(targets, train_preds))
    print("CV ROC-AUC: ", roc_auc_score(targets, oof))

    score = np.zeros(len(test_data))
    for model in tqdm(models):
        score += model.predict_proba(test_data[feat_cols])[:, 1]/len(models)
    submission = pd.DataFrame({
        "id": test_data["id"].values,
        "score": score
    })

    submission.to_csv("submission.csv", index=None)

    return


if (__name__ == "__main__"):
    main()
