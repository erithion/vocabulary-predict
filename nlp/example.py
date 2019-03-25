from sklearn import datasets
from eval_bin_classifier import evaluateOnData

cancer = datasets.load_breast_cancer()

evaluateOnData(cancer.data, cancer.target)