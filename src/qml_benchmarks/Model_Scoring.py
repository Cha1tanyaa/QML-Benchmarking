import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from qml_benchmarks.data import financial_times
from qml_benchmarks.models.my_model import my_model
from qml_benchmarks.models.my_model2 import my_model2
#from qml_benchmarks.models.convolutional_neural_network import ConvolutionalNeuralNetwork

# load data and use labels -1, 1
X, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0, random_state=42)
y = np.array([-1 if y_ == 0 else 1 for y_ in y])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit model
model = my_model()
model2 = my_model2()
#model3 = ConvolutionalNeuralNetwork()

model.fit(X_train, y_train)
model2.fit(X_train, y_train)
#model3.fit(X_train, y_train)

# get predictions
y_pred = model.predict(X_test)
y_pred2 = model2.predict(X_test)

# score the model and print detailed reports
print("Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", model2.score(X_test, y_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))
