import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

# Load the dataset
mobil = pd.read_csv("data/audi.csv")

# Display the first few rows
print(mobil.head())

# Display summary statistics
summary = mobil.describe(include='all')
print(summary)

# Visualize the data
plt.figure(figsize=(12, 6))

# Create boxplot for numeric columns
plt.subplot(1, 2, 1)
sns.boxplot(data=mobil.select_dtypes(include='number'))
plt.title('Boxplot of Numeric Columns')

# Check for null values
print(mobil.isnull().sum())

# Drop duplicates
mobil.drop_duplicates(inplace=True)

# Identify and remove outliers
numeric_cols = mobil.select_dtypes(include=['number']).columns
q1 = mobil[numeric_cols].quantile(0.25)
q3 = mobil[numeric_cols].quantile(0.75)
IQR = q3 - q1

# Identifying outliers
outliers = ((mobil[numeric_cols] < (q1 - 1.5 * IQR)) | (mobil[numeric_cols] > (q3 + 1.5 * IQR))).any(axis=1)
mobil = mobil[~outliers]
print(mobil)

# Identifikasi fitur numerik
numerical_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

# Membuat pipeline untuk pre-processing numerik
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Membagi dataset menjadi fitur dan target
target_column = 'price'
X = mobil[numerical_features]
y = mobil[target_column]

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menerapkan preprocessing pada data numerik
X_train = numerical_transformer.fit_transform(X_train)
X_test = numerical_transformer.transform(X_test)

# Mengecek bentuk dari data yang sudah diproses
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Train the model using Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Print predictions
print("Predictions:", y_pred)

# Calculate and print accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model
joblib.dump(clf, 'decision_tree_model.sav')
