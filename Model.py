import numpy as np
import pickle
from collections import defaultdict

from Eval import Eval
from mapping import drugid2rxnorm, rxnorm2features, id2drug, id2adr
from utils import split_data
from similarity import get_Jaccard_Similarity


class Model:
    def __init__(self, metrics):
        self.ALPHA = 0.1
        self.metrics = metrics

    def get_similarity_matrix(self, X):
        features_matrix = []
        for idx in range(X.shape[0]):
            drug = id2drug.get(idx)
            rxnorm = drugid2rxnorm[drug]
            features = rxnorm2features[rxnorm]
            features_matrix.append(features)
        features_matrix = np.asarray(features_matrix)
        return get_Jaccard_Similarity(features_matrix)

    def label_propogation(self, X, alpha):
        similarity_matrix = self.get_similarity_matrix(X)
        score_matrix_drug = (1 - alpha) * np.matmul(np.linalg.pinv(
            np.eye(np.shape(X)[0]) - alpha * similarity_matrix), X)
        return score_matrix_drug

    def validate(self, X, Y, idx):
        AUC = []
        for i in range(1, 10):
            alpha = i * 0.1
            Y_pred = self.predict(X, alpha)
            metrics = self.eval(Y_pred, Y, idx)
            auc = metrics[0]
            AUC.append(auc)
        print(AUC)
        max_auc = max(AUC)
        max_idx = AUC.index(max_auc)
        max_alpha = (max_idx + 1) * 0.1
        self.ALPHA = max_alpha

    def predict(self, X, alpha):
        Y_pred = self.label_propogation(X, alpha)
        return Y_pred

    def eval(self, Y_pred, Y, idx):
        y_pred, y_gold = [], []
        for r, c in zip(idx[0], idx[1]):
            y_pred.append(Y_pred[r, c])
            y_gold.append(Y[r, c])
        ev = Eval(y_pred, y_gold)
        return ev.Metrics(self.metrics)


    def eval_DME(self, Y_pred, Y, idx, DME):
        y_pred, y_gold = defaultdict(list), defaultdict(list)
        for r, c in zip(idx[0], idx[1]):
            adrid = id2adr.get(c)
            if adrid in DME:
                y_pred[adrid].append(Y_pred[r, c])
                y_gold[adrid].append(Y[r, c])
        EV = {}
        for k in y_pred.keys():
            y_p, y_g = y_pred.get(k), y_gold.get(k)
            ev = Eval(y_p, y_g)
            EV[k] = ev.Metrics(self.metrics)
        return EV




