
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


file_path = "data/adult.data"

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]

df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    skipinitialspace=True
)

print("Dataset loaded successfully!")
print("Dataset size:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nIncome categories:")
print(df["income"].value_counts())

print("\nMissing values in each column:")
print((df == "?").sum())

# Replace missing-value markers with "Unknown"
df = df.replace("?", "Unknown")

print("\nMissing values after cleaning:")
print((df == "?").sum())

print("\nUnknown values after cleaning:")
print((df == "Unknown").sum())

# Step 7: Separate input features and target

X = df.drop("income", axis=1)
y = df["income"].str.strip().str.rstrip(".")

print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget categories:")
print(y.value_counts())

# Step 8: Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# Step 10: Identify numeric and categorical columns

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric columns:")
print(numeric_features)

print("\nCategorical columns:")
print(categorical_features)

# Step 11: Create preprocessing pipelines

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nPreprocessing pipeline created successfully!")

# Step 12: Create and train the model

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model trained successfully!")
# Step 13: Evaluate the model

y_pred = model.predict(X_test)

print("\nModel accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))
# Step 15: Save the trained model

import joblib

joblib.dump(model, "models/income_classifier.pkl")

print("\nModel saved successfully!")
# Step 16: Predict income for a new person

new_person = pd.DataFrame([{
    "age": 30,
    "workclass": "Private",
    "fnlwgt": 150000,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}])

prediction = model.predict(new_person)

print("\nNew person income prediction:")
print(prediction[0])
