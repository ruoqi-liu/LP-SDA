import os
import numpy as np
from tqdm import tqdm

from mapping import sider_eval_pairs, drug2id, adr2id, drug_list, adr_list
class FAERSdata:
    def __init__(self, directory, method, year):

        Files = os.listdir('%s/%s' % (directory, method))

        if year == 'all':
            Files = [Files[-1]]

        X = {}
        Y = {}
        Index = {}
        for i in tqdm(range(len(Files))):
            f = Files[i]
            x = np.zeros(shape=(len(drug_list), len(adr_list)))
            with open('%s/%s/%s' % (directory, method, f), 'r') as ff:
                next(ff)
                for line in ff:
                    line = line.strip('\n')
                    line = line.split(',')
                    drug, adr, score = line[0], line[1], round(float(line[2]),5)
                    drug_id, adr_id = drug2id.get(drug), adr2id.get(adr)
                    if drug in drug_list and adr in adr_list:
                        x[drug_id, adr_id] = score

            y = np.zeros(shape=(len(drug_list), len(adr_list)))
            for drug, adr in sider_eval_pairs:
                drug_id, adr_id = drug2id.get(drug), adr2id.get(adr)
                y[drug_id, adr_id] = 1

            y = np.asarray(y)
            index = np.arange(x.shape[0])

            X[i] = x
            Y[i] = y
            Index[i] = index.tolist()

        self.X = X
        self.Y = Y
        self.Index = Index













