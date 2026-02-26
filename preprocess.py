import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans

# --------------------------------
# STEP 1: Load Dataset
# --------------------------------
data = pd.read_csv("student_dataset.csv.csv")

print("Initial Shape:", data.shape)

# --------------------------------
# STEP 2: Handle GPA Scale
# --------------------------------
if data["Final_CGPA"].max() > 4:
    data["Final_CGPA"] = (data["Final_CGPA"] / 10) * 4
    print("Converted GPA from 10-scale to 4-scale")

# --------------------------------
# STEP 3: Drop Unnecessary Columns
# --------------------------------
if "Student_ID" in data.columns:
    data = data.drop("Student_ID", axis=1)

# --------------------------------
# STEP 4: Handle Missing Values
# --------------------------------
print("\nMissing Values:\n", data.isnull().sum())
data = data.dropna()

# --------------------------------
# STEP 5: Encode Categorical Variables
# --------------------------------
data = pd.get_dummies(data, drop_first=True)

print("\nColumns After Encoding:\n", data.columns)

# --------------------------------
# STEP 6: Separate Features & Target
# --------------------------------
X = data.drop("Final_CGPA", axis=1)
y = data["Final_CGPA"]

# --------------------------------
# STEP 7: Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# STEP 8: Feature Scaling
# --------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------
# STEP 9: Train Advanced Model
# --------------------------------
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# --------------------------------
# STEP 10: Evaluate Model
# --------------------------------
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# --------------------------------
# STEP 11: Clustering (Student Segmentation)
# --------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

data["Cluster"] = clusters

print("\nClustering Completed. Cluster Distribution:")
print(data["Cluster"].value_counts())

# --------------------------------
# STEP 12: Save Everything
# --------------------------------
pickle.dump(model, open("academic_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("model_columns.pkl", "wb"))
pickle.dump(kmeans, open("cluster_model.pkl", "wb"))

print("\nAll models saved successfully.")