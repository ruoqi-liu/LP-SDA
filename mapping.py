import pickle

sider_eval_pairs = pickle.load(open('pickles/sider_eval_pairs_final.pkl', 'rb'))
drugid2rxnorm = pickle.load(open('pickles/drugid2rxnorm_mapping.pkl', 'rb'))
rxnorm2features = pickle.load(open('pickles/rxnorm2features_mapping.pkl', 'rb'))

drug_list = list(set(drug for (drug, adr) in sider_eval_pairs))
adr_list = list(set(adr for (drug, adr) in sider_eval_pairs))

id2drug = {i: drug for i, drug in enumerate(drug_list)}
drug2id = {drug: i for i, drug in enumerate(drug_list)}

id2adr = {i: adr for i, adr in enumerate(adr_list)}
adr2id = {adr: i    for i, adr in enumerate(adr_list)}