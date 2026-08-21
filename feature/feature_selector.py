import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, f_classif


class StableDomainFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        k=8,
        alpha=0.5,
        random_state=None
    ):
        self.k = k
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y, domains=None):
        X = np.asarray(X)
        y = np.asarray(y)
        domains = np.asarray(domains)
        class_score = mutual_info_classif(X, y, random_state=self.random_state)
        domain_variability = self._compute_class_conditional_domain_variability(X, y, domains)

        epsilon = 1e-8
        class_score_norm = class_score / (np.max(class_score) + epsilon)
        variability_norm = domain_variability / (np.max(domain_variability) + epsilon)

        self.class_score_ = class_score
        self.domain_variability_ = domain_variability
        self.final_score_ = class_score_norm - self.alpha * variability_norm

        self.selected_indices_ = np.argsort(self.final_score_)[::-1][:self.k]

        return self


    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_indices_]


    def _compute_class_conditional_domain_variability(self, X, y, domains):
        scores = []
        classes = np.unique(y)
        unique_domains = np.unique(domains)
        for feature_idx in range(X.shape[1]):
            class_variabilities = []
            for cls in classes:
                domain_means = []
                for domain in unique_domains:
                    mask = (y == cls) & (domains == domain)
                    if np.any(mask):
                        domain_means.append(np.mean(X[mask, feature_idx]))
                if len(domain_means) > 1:
                    class_variabilities.append(np.std(domain_means))
            if len(class_variabilities) == 0:
                scores.append(0.0)
            else:
                scores.append(np.mean(class_variabilities))
        return np.asarray(scores)
        