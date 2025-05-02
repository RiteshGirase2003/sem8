# 📦 Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📥 Step 1: Extract - Load the dataset
data = pd.read_csv('car_evaluation.csv', header=None)

# Assign column names
data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

print("Initial Data Preview:")
print(data.head())

# 🔄 Step 2: Transform - Encode categorical features
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
print(data.head())

X = data.drop('class', axis=1)
y = data['class']

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Step 3: Classification - Train a Random Forest Classifier
print("\n Random Forest Classifier")
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators
clf.fit(X_train, y_train)

# 📊 Step 4: Predict & Evaluate
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\n Decision Tree Classifier")


# Classes like 1 and 3 have very few samples compared to class 2 (e.g., 11 vs. 235).
lr = DecisionTreeClassifier(class_weight='balanced')
lr.fit(X_train, y_train)

# 📊 Step 4: Predict & Evaluate
y_pred = lr.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



print("\n Logistic Regression Classifier")


# Classes like 1 and 3 have very few samples compared to class 2 (e.g., 11 vs. 235).
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
lr.fit(X_train, y_train)

# 📊 Step 4: Predict & Evaluate
y_pred = lr.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
