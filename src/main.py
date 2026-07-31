from model import extractor, neutral_feature_builder
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import VARS, numpy as np, os, joblib


def train():
    model = LogisticRegression(
        random_state=VARS.RANDOM_STATE,
        max_iter=VARS.MAX_ITER,
        class_weight='balanced'
    )