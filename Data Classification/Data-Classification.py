# Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
iris = load_iris()

print("DATASET INFORMATION")
print("Dataset Name :", "Iris Dataset")
print("Total Samples :", len(iris.data))
print("Total Features :", len(iris.feature_names))

print("\nFeature Names:")
for feature in iris.feature_names:
    print("-", feature)

print("\nTarget Classes:")
for target in iris.target_names:
    print("-", target)

# Features and Labels
X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDATA SPLIT")
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# Create Model
model = DecisionTreeClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n MODEL ACCURACY")
print("Accuracy :", round(accuracy * 100, 2), "%")

# Predict New Flower

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nNEW PREDICTION")
print("Flower Measurements :", new_flower)
print("Predicted Flower :", iris.target_names[prediction[0]])

# Show First 10 Predictions
print("\nFIRST 10 TEST PREDICTIONS")

for i in range(10):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[predictions[i]]

    print(f"Sample {i+1}")
    print("Actual    :", actual)
    print("Predicted :", predicted)

print("\nProject Completed Successfully!")