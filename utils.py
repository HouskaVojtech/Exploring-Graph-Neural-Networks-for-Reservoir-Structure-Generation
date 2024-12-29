from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from glob import glob
import os.path as osp

import pickle 

def draw_data(u, labels, n_components=3, title='', dot_size=1):
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), s=dot_size)
    if n_components == 2:
        ax = fig.add_subplot(111)
        im = ax.scatter(u[:,0], u[:,1], s=dot_size, c=labels)
        fig.colorbar(im, ax=ax)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(u[:,0], u[:,1], u[:,2], s=dot_size, c=labels)
        fig.colorbar(im, ax=ax)
    plt.title(title, fontsize=18)

def draw_pca(data, labels, n_components=3, title='', dot_size=1):
    fit = PCA(n_components=n_components)
    u = fit.fit_transform(data)
    draw_data(u, labels, n_components, title, dot_size)


def draw_umap(data, labels, n_neighbors=20, min_dist=0.9, n_components=3, metric='euclidean', title='', dot_size=1):
    fit = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    draw_data(u, labels, n_components, title, dot_size)


def draw_tsne(data, labels, n_components=3, perplexity=30, title='', dot_size=1):
    fit = TSNE(n_components=n_components)
    u = fit.fit_transform(data)
    draw_data(u, labels, n_components, title, dot_size)


def get_reservoir_data(file_path):
    with open(file_path, 'rb') as file:
        # Serialize and write the variable to the file
        data = pickle.load(file)
    return data

def load_single_reservoir(file_path, display=False, percentile = 0):
    data = get_reservoir_data(file_path)
    rmse = data['mean_rmse']
    
    if percentile:
        percentile_value = np.percentile(rmse,percentile)

        mask = data['mean_rmse'] < percentile_value
    else:
        mask = np.ones(len(rmse),dtype=bool)
    
    if display:
        sns.histplot(rmse, bins=30, kde=True )
        plt.axvline(percentile_value,color='orange')

        plt.show()
        
    reservoir = np.array(data['reservoir'])[mask] 
    w_in = np.array(data['w_in'])[mask]
    sparsity = np.array(data['sparsity'])[mask] 
    sparsity_mask = np.array(data['sparsity_mask'])[mask]
    feature_vectors = np.array(data['feature_vectors'])[mask] 
    
    return reservoir, w_in, sparsity, sparsity_mask, feature_vectors, rmse

def load_all_reservoirs(root_dir):
    paths = glob(osp.join(root_dir,'*.pickle'))
    all_data = get_reservoir_data(paths[0])
    for idx, path in enumerate(paths[1:]):
        data = get_reservoir_data(path)
        for key in all_data.keys():
            all_data[key] += data[key]
                
    return all_data


