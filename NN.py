# Classification using Neural Networks on Iris dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris.csv")

# Encode target labels
labelencoder = LabelEncoder()
df['target'] = labelencoder.fit_transform(df['target'])

# Extract features and target
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network model
mlp = MLPClassifier(
    hidden_layer_sizes=(12, 12),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=0
)

# Train the model
mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test)

# Evaluate
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print(f"Accuracy : {accuracy:.2f}")
print(cr)

# Display confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=labelencoder.classes_).plot(cmap=plt.cm.Greens)
plt.title("Confusion Matrix")
plt.show()
