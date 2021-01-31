from absl import logging
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from datasets import download_and_prepare
from classifier import SpamClassifier


def main():
    X_train, X_test, y_train, y_test = download_and_prepare("sms_spam", "venv")


    spam_classifier = SpamClassifier()
    spam_classifier.train(X_train, y_train)

    logging.info(f"Train Accuracy: {accuracy_score(y_train, spam_classifier.predict(X_train))}")
    logging.info(f"Test Accuracy: {accuracy_score(y_test, spam_classifier.predict(X_test))}")

    X = np.append(X_train,X_test,axis = 0)
    y = np.append(y_train,y_test,axis = 0)
    accuracy = 0.0
    k = 5
    kf = KFold(n_splits=k, shuffle=True)

    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        spam_classifier.train(train_x, train_y)
        accuracy += accuracy_score(test_y, spam_classifier.predict(test_x))

    accuracy = accuracy/k
    logging.info(f"Test Accuracy_kfold: {accuracy}")


    Bernoulli = BernoulliNB()
    Bernoulli.fit(X_train, y_train)
    logging.info(f"Train Accuracy_bernoulli: {accuracy_score(y_train, Bernoulli.predict(X_train))}")
    logging.info(f"Test Accuracy_bernoulli: {accuracy_score(y_test, Bernoulli.predict(X_test))}")


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    main()