import numpy as np
import random
from sklearn.model_selection import KFold
import pickle
import os
from similarity import get_Jaccard_Similarity

import matplotlib.pyplot as plt

def trian_test_split(y):
    ones_idx = np.where(y == 1)
    zeors_idx = np.where(y == 0)
    n_ones, n_zeors = len(ones_idx[0]), len(zeors_idx[0])
    ones_train_idx = random.sample(range(n_ones), int(0.8 * n_ones))
    zeors_train_idx = random.sample(range(n_zeors), int(0.8 * n_zeors))

    # train
    ones_train_row = ones_idx[0][ones_train_idx]
    zeros_train_row = zeors_idx[0][zeors_train_idx]

    ones_train_col = ones_idx[1][ones_train_idx]
    zeros_train_col = zeors_idx[1][zeors_train_idx]

    train_row = np.append(ones_train_row, zeros_train_row)
    train_col = np.append(ones_train_col, zeros_train_col)

    train_idx = (train_row, train_col)

    # test
    ones_test_idx = [idx for idx in range(n_ones) if idx not in ones_train_idx]
    zeors_test_idx = [idx for idx in range(n_zeors) if idx not in zeors_train_idx]

    ones_test_row = ones_idx[0][ones_test_idx]
    zeros_test_row = zeors_idx[0][zeors_test_idx]

    ones_test_col = ones_idx[1][ones_test_idx]
    zeros_test_col = zeors_idx[1][zeors_test_idx]

    test_row = np.append(ones_test_row, zeros_test_row)
    test_col = np.append(ones_test_col, zeros_test_col)

    test_idx = (test_row, test_col)

    return train_idx, test_idx


def split_data(y):
    ones_idx_r, ones_idx_c = np.where(y==1)
    ones_valid, ones_test = [], []
    kf = KFold(n_splits=5)
    for valid_idx, test_idx in kf.split(ones_idx_r):
        ones_valid_row = ones_idx_r[valid_idx]
        ones_valid_col = ones_idx_c[valid_idx]
        ones_test_row = ones_idx_r[test_idx]
        ones_test_col = ones_idx_c[test_idx]

        ones_valid = (ones_valid_row, ones_valid_col)
        ones_test = (ones_test_row, ones_test_col)
        break

    zeros_idx_r, zeros_idx_c = np.where(y == 0)
    zeros_valid, zeros_test = [], []
    for valid_idx, test_idx in kf.split(zeros_idx_r):
        zeros_valid_row = zeros_idx_r[valid_idx]
        zeros_valid_col = zeros_idx_c[valid_idx]
        zeros_test_row = zeros_idx_r[test_idx]
        zeros_test_col = zeros_idx_c[test_idx]

        zeros_valid = (zeros_valid_row, zeros_valid_col)
        zeros_test = (zeros_test_row, zeros_test_col)
        break

    valid = (np.append(ones_valid[0], zeros_valid[0]), np.append(ones_valid[1], zeros_valid[1]))
    test = (np.append(ones_test[0], zeros_test[0]), np.append(ones_test[1], zeros_test[1]))

    return valid, test


def sample_zeros(y):
    zeros_r_idx, zeros_c_idx = np.where(y==0)
    ones_r_idx, ones_c_idx = np.where(y==1)

    random.seed(16)

    sample_zeros_pos = random.sample(range(len(zeros_r_idx)), len(ones_r_idx)*2)
    sample_zeros_r_idx, sample_zeros_c_idx = zeros_r_idx[sample_zeros_pos], zeros_c_idx[sample_zeros_pos]

    sample_r_idx = np.append(ones_r_idx, sample_zeros_r_idx)
    sample_c_idx = np.append(ones_c_idx, sample_zeros_c_idx)

    return (sample_r_idx, sample_c_idx)

def curate_cid(cid):
    cid = cid[3:]
    for i, d in enumerate(cid):
        if d == '0':
            continue
        else:
            return cid[i:]

def get_sider_pairs(dump_file):
    cid2rxnorm_mapping = pickle.load(open('pickles/cid2rxnorm_mapping.pkl', 'rb'))
    rxnorm2features_mapping = pickle.load(open('pickles/rxnorm2features_mapping.pkl', 'rb'))
    rxnorm2drugid_mapping = pickle.load(open('pickles/rxnorm2drugid_mapping.pkl', 'rb'))
    adrlist_freq = pickle.load(open('pickles/adrlist_freq.pkl', 'rb'))
    umlid2adrid_mapping = pickle.load(open('pickles/umlid2adrid_mapping.pkl', 'rb'))

    sider_eval_id_label = set()
    with open('OFFSIDE/meddra_all_se.tsv', 'r') as f:
        next(f)
        for row in f:
            row = row.strip('\n')
            row = row.split('\t')
            cid, adr_type, umlid, adr = row[1], row[3], row[4], row[5]
            cid = curate_cid(cid)
            if umlid in umlid2adrid_mapping:
                adrid = umlid2adrid_mapping[umlid]
                if cid in cid2rxnorm_mapping and adrid in adrlist_freq and adr_type == 'PT':
                    rxnorm = cid2rxnorm_mapping[cid]
                    if rxnorm in rxnorm2features_mapping:
                        drugid = rxnorm2drugid_mapping[rxnorm]
                        sider_eval_id_label.add((drugid, adrid))

    pickle.dump(sider_eval_id_label, open(dump_file, 'wb'))

    drug_list = list(set(drug for (drug, adr) in sider_eval_id_label))
    adr_list = list(set(adr for (drug, adr) in sider_eval_id_label))
    print('n_drug: {}'.format(len(drug_list)))
    print('n_adr: {}'.format(len(adr_list)))



def get_offside_pairs(dump_file):
    cid2rxnorm_mapping = pickle.load(open('pickles/cid2rxnorm_mapping.pkl', 'rb'))
    rxnorm2features_mapping = pickle.load(open('pickles/rxnorm2features_mapping.pkl', 'rb'))
    rxnorm2drugid_mapping = pickle.load(open('pickles/rxnorm2drugid_mapping.pkl', 'rb'))
    adrlist_freq = pickle.load(open('pickles/adrlist_freq.pkl', 'rb'))[:1000]
    umlid2adrid_mapping = pickle.load(open('pickles/umlid2adrid_mapping.pkl', 'rb'))

    sider_eval_id_label = set()
    with open('OFFSIDE/3003377s-offsides.tsv', 'r') as f:
        next(f)
        for row in f:
            row = row.strip('\n')
            row = row.split('\t')
            cid, umlid = row[0].strip('\"'), row[2].strip('\"')
            cid = curate_cid(cid)
            if umlid in umlid2adrid_mapping:
                adrid = umlid2adrid_mapping[umlid]
                if cid in cid2rxnorm_mapping and adrid in adrlist_freq:
                    rxnorm = cid2rxnorm_mapping[cid]
                    if rxnorm in rxnorm2features_mapping:
                        drugid = rxnorm2drugid_mapping[rxnorm]
                        sider_eval_id_label.add((drugid, adrid))

    pickle.dump(sider_eval_id_label, open(dump_file, 'wb'))
    drug_list = list(set(drug for (drug, adr) in sider_eval_id_label))
    adr_list = list(set(adr for (drug, adr) in sider_eval_id_label))
    print('n_drug: {}'.format(len(drug_list)))
    print('n_adr: {}'.format(len(adr_list)))


def construct_sider_signal_scores_source(source_file, signal_file, method, sider_eval_pairs):
    # load mappings
    drugid2rxnorm_mapping = pickle.load(open('pickles/drugid2rxnorm_mapping.pkl', 'rb'))
    rxnorm2cid_mapping = pickle.load(open('pickles/rxnorm2cid_mapping.pkl', 'rb'))
    adrids = set([adrid for (drugid, adrid) in sider_eval_pairs])
    drugids = set([drugid for (drugid, adrid) in sider_eval_pairs])

    for year in ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']:
        print('process years: {}*********'.format(year))
        original_signal_scores_path = source_file + method + '/' + method + '_' + year + '.csv'
        sider_signal_scores_path = signal_file + method + '/' + method + '_' + year + '.csv'
        os.makedirs(os.path.dirname(sider_signal_scores_path), exist_ok=True)
        out = open(sider_signal_scores_path, 'w')
        drug_adr_scores_pair = {}
        drug = set()
        with open(original_signal_scores_path, 'r') as f:
            next(f)
            for row in f:
                row = row.strip('\n')
                row = row.split(',')
                drugid, adrid, score = row[0], row[1], row[2]
                if score == 'Inf' or score == 'NaN':
                    score = 0
                if drugid in drugids and adrid in adrids:
                    drug_adr_scores_pair[(drugid, adrid)] = score
                    drug.add(drugid)

        for (d_id, a_id) in sider_eval_pairs:
            rxnorm = drugid2rxnorm_mapping[d_id]
            if rxnorm in rxnorm2cid_mapping:
                if (d_id, a_id) not in drug_adr_scores_pair:
                    drug_adr_scores_pair[(d_id, a_id)] = 0


        drug_list = set()
        adr_list = set()

        n_pair = 0
        for (drugid, adrid) in drug_adr_scores_pair.keys():
            score = drug_adr_scores_pair[(drugid, adrid)]
            drug_list.add(drugid)
            adr_list.add(adrid)
            if score != 0:
                n_pair += 1
            out.write(drugid + ',' + adrid + ',' + str(score) + '\n')



        print('{} drugs found with drug features info.'.format(len(drug_list)))
        print('{} adrs found with drug features info.'.format(len(adr_list)))
        print('{} pairs found with drug features info.'.format(n_pair))

        out.close()


def get_similarity_score(drugid):
    drugid2rxnorm = pickle.load(open('pickles/drugid2rxnorm_mapping.pkl', 'rb'))
    rxnorm2features = pickle.load(open('pickles/rxnorm2features_mapping.pkl', 'rb'))

    features_matrix = []
    drug_list = []
    for did in drugid2rxnorm.keys():
        rxnorm = drugid2rxnorm.get(did)
        if rxnorm in rxnorm2features:
            drug_list.append(did)
            features_matrix.append(rxnorm2features.get(rxnorm))

    features_matrix = np.asarray(features_matrix)
    similarity_matrix = get_Jaccard_Similarity(features_matrix)

    drug_idx = drug_list.index(drugid)

    similarity_score = similarity_matrix[drug_idx].tolist()[0]

    out = open('res/9.26/' + drugid + '.csv', 'w')
    for i in range(len(drug_list)):
        out.write(drug_list[i] + ',' + str(similarity_score[i]) + '\n')
    out.close()


def get_drug_labels(adrnames):
    cid2rxnorm_mapping = pickle.load(open('pickles/cid2rxnorm_mapping.pkl', 'rb'))
    rxnorm2features_mapping = pickle.load(open('pickles/rxnorm2features_mapping.pkl', 'rb'))
    rxnorm2drugid_mapping = pickle.load(open('pickles/rxnorm2drugid_mapping.pkl', 'rb'))
    adrlist_freq = pickle.load(open('pickles/adrlist_freq.pkl', 'rb'))
    umlid2adrid_mapping = pickle.load(open('pickles/umlid2adrid_mapping.pkl', 'rb'))
    adrname2id = np.loadtxt('data/adrid_name.csv', delimiter='$', dtype=str, usecols=(0,1))
    adrname2id_mappiing = {adrname: adrid for [adrid, adrname] in adrname2id}
    adrids = set(adrname2id_mappiing.get(name) for name in adrnames if name in adrname2id_mappiing)

    drug_labels = set()
    with open('OFFSIDE/meddra_all_se.tsv', 'r') as f:
        next(f)
        for row in f:
            row = row.strip('\n')
            row = row.split('\t')
            cid, adr_type, umlid, adr = row[1], row[3], row[4], row[5]
            cid = curate_cid(cid)
            if umlid in umlid2adrid_mapping:
                adrid = umlid2adrid_mapping[umlid]
                if cid in cid2rxnorm_mapping and adrid in adrlist_freq and adr_type == 'PT':
                    rxnorm = cid2rxnorm_mapping[cid]
                    if rxnorm in rxnorm2features_mapping:
                        drugid = rxnorm2drugid_mapping[rxnorm]
                        if adrid in adrids:
                            drug_labels.add(drugid)

    return drug_labels




def plot_parameter():
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prr = [0.7246079585714792, 0.7264323654528433, 0.7280536464436141, 0.7293487502200561, 0.7299513736139125, 0.7291259449336482, 0.7253171549678908, 0.7157189870841383, 0.6952790118859359]
    ror = [0.7232861946865585, 0.7250917752194431, 0.726693325051255, 0.7279560375177935, 0.7285419746094042, 0.7277724794885267, 0.7241928580200252, 0.7149674790756472, 0.6949297556790539]


    l1 = plt.plot(alpha, prr, '')







