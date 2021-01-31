from absl import logging
import time
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

class SpamClassifier(object):
    """
    Spam classifier using (multinomial) Naive Bayes

    Parameters:
        alpha (float): Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    """
    def __init__(self, alpha=1.0):
        super(SpamClassifier, self).__init__()
        self.alpha = alpha

    def train(self, X, y):
        """
        Training method

        Estimates the log-likelihoods and priors for both classes ham and spam.

        Parameters:
            X (ndarray): Feature matrix with shape (num_samples, num_features)
            y (ndarray): Label vector with shape (num_samples,)
        """
        logging.info(f"Starting training...")
        start_time = time.time()

        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)


        self.priors_spam = sum(y)/len(y)
        self.priors_ham = 1 - self.priors_spam
        # TODO: Estimate priors
        self.log_priors_spam = np.log(self.priors_spam)
        self.log_priors_ham = np.log(self.priors_ham)

        self.spam_vector = np.ones(n_features)*self.alpha
        self.ham_vector = np.ones(n_features)*self.alpha
        self.N_vocab_spam = self.alpha*n_features
        self.N_vocab_ham = self.alpha*n_features
        for i in range(n_samples):
            if y[i] == 1:
                self.spam_vector += X[i]
                self.N_vocab_spam += sum(X[i])
            else:
                self.ham_vector += X[i]
                self.N_vocab_ham += sum(X[i])
        self.log_prob_spam = np.log(self.spam_vector / self.N_vocab_spam)
        self.log_prob_ham = np.log(self.ham_vector / self.N_vocab_ham)

        # TODO: Estimate log-likelihoods

        logging.debug(f"Training took {int(time.time() - start_time)} seconds.")


    def predict(self, X):
        """
        Prediction method

        Uses Bayes rule to compute un-normalized posteriors

        Parameters:
            X (ndarray): Feature matrix with shape (num_samples, num_features)

        Returns:
            (ndarray): Prediction vector with shape (num_samples,)
        """
        # TODO: Implement MAP decision for multinomial Naive Bayes
        label = []
        n_samples, n_features = X.shape
        for i in range(n_samples):
            P_ham = sum(X[i]*self.log_prob_ham) + self.log_priors_ham
            P_spam = sum(X[i]*self.log_prob_spam) + self.log_priors_spam

            if P_spam >= P_ham:
                label.append(1)
            else:
                label.append(0)
        label = np.array(label)
        return label


        






