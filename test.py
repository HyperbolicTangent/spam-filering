import numpy as np

# int array
array1 = np.array([1,0,1,0])
# convert int array to bool list
list1 = [True if array1[i]==1 else False for i in range(len(array1))]
# convert bool list to bool array
array1 = np.array(list1)

print(array1)


def train_bernoulli(self, X, y):
    n_samples, n_features = X.shape
    self.classes = np.unique(y)
    self.n_classes = len(self.classes)

    # self.priors = np.ones((self.n_classes,)) / self.n_classes # This is just a placeholder
    self.priors_spam = sum(y) / len(y)
    self.priors_ham = 1 - self.priors_spam

    # self.log_priors = np.log(self.priors)
    self.log_priors_spam = np.log(self.priors_spam)
    self.log_priors_ham = np.log(self.priors_ham)
    self.spam_vector = np.zeros(n_features)
    self.ham_vector = np.zeros(n_features)

    for i in range(n_samples):
        if y[i] == 1:
            self.spam_vector += X[i]
            # self.N_vocab_spam += sum(X[i])
        else:
            self.ham_vector += X[i]
            # self.N_vocab_ham += sum(X[i])
    self.spam_vector_ = [1 if self.spam_vector[i] > 0 else 0 for i in range(n_features)]
    self.ham_vector_ = [1 if self.ham_vector[i] > 0 else 0 for i in range(n_features)]
    self.spam_vector = np.array(self.spam_vector)
    self.ham_vector = np.array(self.ham_vector)
    self.N_vocab_spam = sum(self.spam_vector_) + 2
    self.N_vocab_ham = sum(self.ham_vector_) + 2
    self.po_log_prob_spam = np.log((self.spam_vector + 1) / self.N_vocab_spam)
    self.po_log_prob_ham = np.log((self.ham_vector + 1) / self.N_vocab_ham)
    self.neg_log_prob_spam = np.log(1 - np.exp(self.po_log_prob_spam))
    self.neg_log_prob_ham = np.log(1 - np.exp(self.po_log_prob_ham))
    # self.log_prob_spam = self.po_log_prob_spam - self.neg_log_prob_spam
    # self.log_prob_ham = self.po_log_prob_ham - self.neg_log_prob_ham

    def predict_bernoulli(self, X):
        label = []
        n_samples, n_features = X.shape
        P_ham = 0
        P_spam = 0
        for i in range(n_samples):
            X[i] = [1 if X[i][j] >0 else 0 for j in range(n_features)]
            X[i] = np.array(X[i])
            P_ham = sum(X[i]*(self.po_log_prob_ham - self.neg_log_prob_ham))
            P_spam = sum(X[i]*(self.po_log_prob_spam - self.neg_log_prob_spam))


            P_ham += self.log_priors_ham + sum(self.neg_log_prob_ham)
            P_spam += self.log_priors_spam + sum(self.neg_log_prob_spam)
            #P_ham = sum(X[i]*self.log_prob_ham) + self.log_priors_ham
            #P_spam = sum(X[i]*self.log_prob_spam) + self.log_priors_spam

            if P_spam >= P_ham:
                label.append(1)
            else:
                label.append(0)
        label = np.array(label)
        return label