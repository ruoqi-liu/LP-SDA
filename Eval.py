from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, precision_recall_curve
import numpy as np

class Eval:
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold

    def Metrics(self, metics):
        if metics == 'all':
            auc = roc_auc_score(self.gold, self.pred)
            aupr = average_precision_score(self.gold, self.pred)
            recalls, precisions, thresholds_pr = precision_recall_curve(self.gold, self.pred)
            f1s = (2 * np.multiply(precisions, recalls)) / np.add(precisions, recalls)
            f1s = np.nan_to_num(f1s)
            max_idx = int(np.argmax(f1s))
            precision = precisions[max_idx]
            recall = recalls[max_idx]
            f1 = f1s[max_idx]
            threshold = thresholds_pr[max_idx]
            y_scores_label = np.copy(self.pred)
            y_scores_label = np.where(y_scores_label > threshold, 1, 0)
            y_scores_label = y_scores_label.astype(int)
            accuracy = accuracy_score(self.gold, y_scores_label)
            return np.array([auc, aupr, precision, recall, accuracy, f1])
        elif metics == 'specificity-sensitivity':
            # recall, sensitivity: true positive rate tp/(tp+fn)
            # specificity: true negative rate tn/(tn+fp)
            auc = roc_auc_score(self.gold, self.pred)

            fixed_sensitivity, fixed_specificity = [], []
            fpr, tpr, _ = roc_curve(self.gold, self.pred)
            sensitivity, specificity = tpr, 1 - fpr

            for i in range(1, 10):
                value = i * 0.1
                sensitivity_idx = np.argmin(np.abs(sensitivity-value))
                spec = specificity[sensitivity_idx]
                fixed_sensitivity.append(spec)

                specificity_idx = np.argmin(np.abs(specificity-value))
                sen = sensitivity[specificity_idx]
                fixed_specificity.append(sen)

            return auc, np.array(fixed_sensitivity), np.array(fixed_specificity)
