import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

train_label_file = ""
while train_label_file != "0" and train_label_file != "1":
    train_label_file = input("0 = True Label; 1 = False Label")

if train_label_file == "0":
    train_label_file = "y_train.txt"
    print("Using clean Labels")
else:
    train_label_file = "y_train - Poisoned.txt"
    print("Using poisoned Labels")

# Load the WiLI-2018 dataset 
with open('x_train.txt', 'r', encoding='utf-8') as file:
    x_train_lines = file.readlines()

x_train_data = [line.strip() for line in x_train_lines]

print("read training data")
# Labels
with open(train_label_file, 'r', encoding='utf-8') as file:
    y_train_lines = file.readlines()

y_train_data = [line.strip() for line in y_train_lines]
print("read labels")

# Input / zu vorhersagender Satz
with open('x_test.txt', 'r', encoding='utf-8') as file:
    x_test_lines = file.readlines()

x_test_data = [line.strip() for line in x_test_lines]

with open('y_test.txt', 'r', encoding='utf-8') as file:
    y_test_lines = file.readlines()

y_test_data = [line.strip() for line in y_test_lines]

print("read test data")


# Extracting n-gram features by letter and tokenizing
ngram_range = (2, 3)  
vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
X_train = vectorizer.fit_transform(x_train_data)
X_test = vectorizer.transform(x_test_data)

print("finished vectorizing")

# Training
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train_data)

print("finished training")


# True labels for test data
true_labels = y_test_data 

# Transform the test data using the same vectorizer
X_test_transformed = vectorizer.transform(x_test_data)
# Calculate predictions
predictions = naive_bayes.predict(X_test_transformed)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

# Calculate recall, and F1-score
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

print("Recall:", recall)
print("F1 Score:", f1)

#Plotting of scores
x = np.array(["ACC","REC", "F1"])
y = np.array([accuracy, recall, f1])

plt.bar(x,y)
plt.show()

# Print classification report
#print(classification_report(true_labels, predictions))