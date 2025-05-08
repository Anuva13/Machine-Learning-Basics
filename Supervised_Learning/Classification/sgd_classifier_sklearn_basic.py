from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
