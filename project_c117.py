#CONFUSION MATRIX-PROJECT C117 SOUMILI DEY

from google.colab import files
data_to_load= files.upload()

import pandas as pd
df = pd.read_csv("BankNote_Authentication.csv")

from sklearn.model_selection import train_test_split
variance = df["variance"]
class1 = df["class"]

variance_train, variance_test, class1_train, class1_test = train_test_split(variance, class1, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.reshape(variance_train.ravel(), (len(variance_train), 1))
Y = np.reshape(class1_train.ravel(),(len(class1_train), 1))

classifier = LogisticRegression(random_state = 0)
classifier.fit(X,Y)

X_test = np.reshape(variance_test.ravel(),(len(variance_test), 1))
Y_test = np.reshape(class1_test.ravel(),(len(class1_test), 1))

Class_prediction = classifier.predict(X_test)

predicted_values = []

for i in Class_prediction:
  if i == 0:
    predicted_values.append("Authorized")
  else:
    predicted_values.append("Forged")

actual_values = []

for i in Y_test.ravel():
  if i == 0:
    actual_values.append("Authorized")
  else:
    actual_values.append("Forged")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

labels = ["Forged", "Authorized"]

ax = plt.subplot()
cm = confusion_matrix(actual_values, predicted_values)
sns.heatmap(cm, annot= True, ax=ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

accuracy = (1.7e+02+ 1.2e+02 )/(1.7e+02 + 27 + 29 +1.2e+02 )
print(accuracy)
