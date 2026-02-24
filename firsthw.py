# Machine Learning Homework 1
# Iris Dataset - KNeighborsClassifier

#1 Import libraries and load the dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()

#2 Display all features of the dataset
print("Dataset shape:", iris.data.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Unique target values:", set(iris.target))
print("\nDataset description:")
print(iris.DESCR[:500])

#3 Convert dataset into a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

#4 Display head and tail of the dataset
print("\nHead (first 5 rows):")
print(df.head())

print("\nTail (last 5 rows):")
print(df.tail())

#5 Visualize the dataset
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# scatter plot for sepal features
colors = ['red', 'green', 'blue']
for i in range(3):
    mask = df['target'] == i
    axes[0].scatter(
        df[mask]['sepal length (cm)'],
        df[mask]['sepal width (cm)'],
        c=colors[i],
        label=iris.target_names[i]
    )
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')
axes[0].set_title('Iris Dataset - Sepal')
axes[0].legend()

# scatter plot for petal features
for i in range(3):
    mask = df['target'] == i
    axes[1].scatter(
        df[mask]['petal length (cm)'],
        df[mask]['petal width (cm)'],
        c=colors[i],
        label=iris.target_names[i]
    )
axes[1].set_xlabel('Petal Length')
axes[1].set_ylabel('Petal Width')
axes[1].set_title('Iris Dataset - Petal')
axes[1].legend()

plt.tight_layout()
plt.savefig('iris_visualization.png')
print("\nPlot saved as iris_visualization.png")

#6 Classification using KNeighborsClassifier (default k=5)
# split data into features (X) and target (y)
X = iris.data
y = iris.target

# split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# create model with default k value (k=5)
model = KNeighborsClassifier()

# train the model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

#7 Prediction accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel predictions: {y_pred}")
print(f"Actual values:     {y_test}")
print(f"\n*** Model Accuracy: {accuracy * 100:.2f}% ***")