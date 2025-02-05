# /bin/python
from icecream import ic
def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression()
	# cls = LogisticRegression(penalty = "elasticnet", l1_ratio = 0.4, solver = "saga")
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	ic("  Accuracy", acc)



