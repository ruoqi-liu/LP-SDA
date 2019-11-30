import numpy as np


def get_Euclidean_Similarity(interaction_matrix):
    X=np.mat(interaction_matrix)
    row_matrix=np.power(interaction_matrix,2).sum(axis=1)
    distance_matrix=row_matrix+row_matrix.T-2*np.dot(X,X.T)
    distance_matrix=np.sqrt(distance_matrix)
    ones_matrix = np.ones(distance_matrix.shape)
    similarity_matrix=np.divide(np.mat(ones_matrix),(distance_matrix+ones_matrix))

    # similarity_matrix=np.divide(ones_matrix,(np.exp(-similarity_matrix)+ones_matrix))
    # for i in range(similarity_matrix.shape[0]):
    #     similarity_matrix[i, i] = 0
    return matrix_normalize(similarity_matrix)



def get_Jaccard_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    E = np.ones_like(X.T)
    denominator=X * E + E.T * X.T - X * X.T
    denominator_zero_index=np.where(denominator==0)
    denominator[denominator_zero_index]=1
    result = X * X.T / denominator
    result[denominator_zero_index]=0
    result = result - np.diag(np.diag(result))

    return matrix_normalize(result)


def get_Cosin_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = np.multiply(X, X).sum(axis=1)
    similarity_matrix = X * X.T / (np.sqrt(alpha * alpha.T))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    for i in range(similarity_matrix.shape[0]):
        similarity_matrix[i, i] = 0

    return matrix_normalize(similarity_matrix)



def get_Pearson_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    X = X - (np.divide(X.sum(axis=1),X.shape[1]))
    similarity_matrix = get_Cosin_Similarity(X)
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    for i in range(similarity_matrix.shape[0]):
        similarity_matrix[i, i] = 0
    return similarity_matrix


def get_Gauss_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    delta = 1 / np.mean(np.power(X,2), 0).sum()
    alpha = np.power(X, 2).sum(axis=1)
    result = np.exp(np.multiply(-delta, alpha + alpha.T - 2 * X * X.T))
    result[np.isnan(result)] = 0
    result = result - np.diag(np.diag(result))

    return matrix_normalize(result)


def matrix_normalize(similarity_matrix):
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        for i in range(similarity_matrix.shape[0]):
            similarity_matrix[i, i] = 0
        for i in range(50):
            D = np.diag(np.array(np.sum(similarity_matrix, axis=1)).flatten())
            D= np.divide(np.ones(similarity_matrix.shape),np.sqrt(D))
            D[np.isinf(D)]=0
            similarity_matrix = D * similarity_matrix * D
    else:
        for i in range(similarity_matrix.shape[0]):
            if np.sum(similarity_matrix[i], axis=1) == 0:
                similarity_matrix[i] = similarity_matrix[i]
            else:
                similarity_matrix[i] = similarity_matrix[i] / np.sum(similarity_matrix[i], axis=1)
    return similarity_matrix




def fast_LNS_similarity(feature_matrix, neighbor_num):
    iteration_max = 50
    mu = 6
    X = np.mat(feature_matrix)
    vector_sqr = np.power(X, 2).sum(axis=1)
    distance_matrix = vector_sqr + vector_sqr.T - 2 * X * X.T
    row_num = X.shape[0]
    e = np.ones((row_num,1))
    distance_matrix = np.array(distance_matrix+np.diag(np.diag(e*e.T*np.inf)))
    sort_index = np.argsort(distance_matrix, kind='mergesort')
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num,row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    np.random.seed(0)
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)
    lamda = mu * e
    P = X * X.T + lamda * e.T
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W)
    return np.array(W)