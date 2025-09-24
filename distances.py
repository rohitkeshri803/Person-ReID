import torch
import torch.nn.functional as F


def mahalanobis_dist_from_features(outputs):
    batch_size = outputs.size(0)

    # Compute Mahalanobis distance matrix.
    magnitude = (outputs ** 2).sum(1).expand(batch_size, batch_size)
    squared_matrix = outputs.mm(torch.t(outputs))
    mahalanobis_distances = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt()
    return mahalanobis_distances

def mahalanobis_dist_from_vectors(x, y):
    
    X = torch.cat((x, y))
    X = mahalanobis_dist_from_features(X)
    #print(X)
    return X[0][1]                              #bad coding but still

def calculate_similarity(features):                   #features->concatenate query and g features   
    x = mahalanobis_dist_from_features(features)
    #print(x)
    #print("*************************************************")
    x = x/torch.max(x)
    #print(x)
    #print("*************************************************")
    matrix = 1 - x
    #print(matrix)
    matrix.fill_diagonal_(0)                                       #look into this
    return matrix