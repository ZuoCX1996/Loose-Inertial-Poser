import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class DataVisualizer():
    def __init__(self, method='pca'):
        if method == 'pca':
            self.enbeder = PCA(n_components=2)
        elif method == 'tsne':
            self.enbeder = TSNE(n_components=2, init='pca')
            self.tsne_pca = PCA(n_components=10)
        else:
            print("method='pca' or 'tsne'")
            return
        self.method = method
        self.scaler = StandardScaler()

    def fit_transform(self, X, data_col, class_col, sample_rate=None):
        if sample_rate is not None:
            X = X.sample(frac=sample_rate, random_state=42).reset_index()
        X_std = self.scaler.fit_transform(X[data_col])
        y = X[class_col].values
        if self.method == 'pca':
            X_pca = self.enbeder.fit_transform(X_std)
            X_result = np.hstack([X_pca, y])
        elif self.method == 'tsne':
            X_std = self.tsne_pca.fit_transform(X_std)
            X_tsne = self.enbeder.fit_transform(X_std)
            X_result = np.hstack([X_tsne, y])

        df_result = pd.DataFrame(X_result, columns=['Dim1', 'Dim2'] + class_col)
        df_result.head()
        plt.figure(figsize=(8, 8))

        dict_color = {0: 'r', 1: 'orange', 2: 'lime', 3: 'aqua', 4: 'b', 5: 'fuchsia'}
        # sns.scatterplot(data=df_tsne, hue=class_col[0], x='Dim1', y='Dim2', palette=dict_color)
        sns.scatterplot(data=df_result, hue=class_col[0], x='Dim1', y='Dim2', sizes=0.1)
        # plt.xlim(0, 8)
        # plt.ylim(-4, 4)
        plt.show()

        return df_result

    def transform(self, X, data_col, class_col, sample_rate=None):
        if sample_rate is not None:
            X = X.sample(frac=sample_rate, random_state=42).reset_index()
        X_std = self.scaler.transform(X[data_col])
        y = X[class_col].values
        if self.method == 'pca':
            X_pca = self.enbeder.transform(X_std)
            X_result = np.hstack([X_pca, y])
        elif self.method == 'tsne':
            X_std = self.tsne_pca.transform(X_std)
            X_tsne = self.enbeder.fit_transform(X_std)
            X_result = np.hstack([X_tsne, y])

        df_result = pd.DataFrame(X_result, columns=['Dim1', 'Dim2'] + class_col)
        df_result.head()
        plt.figure(figsize=(8, 8))

        dict_color = {0: 'r', 1: 'orange', 2: 'lime', 3: 'aqua', 4: 'b', 5: 'fuchsia'}
        # sns.scatterplot(data=df_tsne, hue=class_col[0], x='Dim1', y='Dim2', palette=dict_color)
        sns.scatterplot(data=df_result, hue=class_col[0], x='Dim1', y='Dim2', sizes=0.1)
        # plt.xlim(-10, 15)
        # plt.ylim(-10, 16)
        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        plt.show()

        return df_result

