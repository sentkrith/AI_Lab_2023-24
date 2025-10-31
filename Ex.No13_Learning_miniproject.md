# Ex.No: 13 Learning – Use Supervised Learning  
### DATE: 25/10/2025                                                                       
### REGISTER NUMBER : 212223060254
### AIM: 
The aim of this model is to predict the risk of liver failure using key clinical and biochemical parameters through machine learning. It helps in early detection of liver disorders and supports data-driven medical decision-making.

##  **Objective**

The objective of this model is to **develop an efficient and accurate machine learning system** that can:

1. **Predict the presence or risk of liver failure** based on patient biochemical and demographic data.
2. **Optimize model performance** through data preprocessing and hyperparameter tuning using GridSearchCV.
3. **Support early diagnosis and clinical decision-making** by identifying key indicators linked to liver dysfunction.
4. **Provide a reusable predictive framework** for integration into healthcare analytics or digital diagnostic tools.

### Introduction:
This project focuses on predicting liver failure using machine learning techniques based on patient biochemical and demographic data. It aims to enable early detection of liver disorders and assist healthcare professionals in making accurate, data-driven decisions.

###  Algorithm:
1. **Data Preparation:** Load or generate the liver dataset, handle missing values, and encode categorical features like Gender.
2. **Preprocessing:** Apply scaling for numeric features and one-hot encoding for categorical features using a `ColumnTransformer`.
3. **Model Training & Tuning:** Train a `RandomForestClassifier` combined with `GridSearchCV` to find the best hyperparameters.
4. **Evaluation & Prediction:** Evaluate model performance using accuracy, precision, recall, and F1-score, then save the best model for future predictions.

### Program:
```python
# Load dataset (prefer the updated CSV with Gender as 'M'/'F')
csv_path = R"C:\Users\acer\Desktop\Anaconda\data\liver_synthetic_gender_str.csv.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path}")
else:
    print(f"{csv_path} not found — generating synthetic dataset now.")
    def create_synthetic_liver_data_gender_str(n_samples=300, random_state=2025):
        rng = np.random.RandomState(random_state)
        age = rng.randint(20, 86, size=n_samples)
        gender_binary = rng.binomial(1, 0.6, size=n_samples)
        gender = np.where(gender_binary == 1, 'M', 'F')
        total_bilirubin = np.round(np.abs(rng.normal(1.2, 1.5, size=n_samples)), 2)
        direct_bilirubin = np.round(total_bilirubin * rng.uniform(0.1, 0.7, size=n_samples), 2)
        alkaline_phosphatase = np.round(rng.normal(100, 40, size=n_samples).clip(30, 400), 0)
        alt = np.round(rng.normal(35, 50, size=n_samples).clip(5, 800), 0)
        ast = np.round(rng.normal(40, 45, size=n_samples).clip(5, 700), 0)
        total_proteins = np.round(rng.normal(7.0, 0.7, size=n_samples), 2).clip(4.0, 9.0)
        albumin = np.round(rng.normal(4.0, 0.5, size=n_samples), 2).clip(2.0, 5.5)
        ag_ratio = np.round(rng.normal(1.1, 0.4, size=n_samples), 2).clip(0.4, 2.5)
        risk_score = (
            0.6 * total_bilirubin + 0.4 * direct_bilirubin + 0.003 * alkaline_phosphatase
            + 0.004 * alt + 0.004 * ast - 0.5 * albumin - 0.2 * total_proteins
            + 0.1 * (age / 50.0) + 0.1 * gender_binary + rng.normal(0, 0.5, size=n_samples)
        )
        threshold = np.percentile(risk_score, 70)
        liver_disease = (risk_score > threshold).astype(int)
        df_local = pd.DataFrame({
            'Age': age, 'Gender': gender, 'Total_Bilirubin': total_bilirubin,
            'Direct_Bilirubin': direct_bilirubin, 'Alkaline_Phosphotase': alkaline_phosphatase,
            'ALT': alt, 'AST': ast, 'Total_Proteins': total_proteins,
            'Albumin': albumin, 'A_G_Ratio': ag_ratio, 'Liver_Disease': liver_disease
        })
        for col in ['A_G_Ratio', 'Albumin', 'Total_Proteins']:
            mask = rng.rand(n_samples) < 0.05
            df_local.loc[mask, col] = np.nan
        return df_local
    df = create_synthetic_liver_data_gender_str(n_samples=300, random_state=2025)
    df.to_csv(csv_path, index=False)
    print(f"Saved generated dataset to {csv_path}")

print('\nDataset shape:', df.shape)
df.head()
# ---------------------------
# Preprocessing setup
# ---------------------------

X = df.drop(columns=['Liver_Disease']).copy()
y = df['Liver_Disease'].copy()

# Detect numeric and categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Changed 'sparse' to 'sparse_output'
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
# -----------------------------
# Combine RandomForest + GridSearchCV
# -----------------------------

# Pipeline using preprocessing + RandomForest
pipeline_rf = Pipeline([
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Parameter grid for tuning
param_grid = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 6, 12],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}

gs = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=2
)

start = time.time()
gs.fit(X_train, y_train)
end = time.time()

print(f"\nGridSearch completed in {(end - start):.1f} seconds")
print("Best params:", gs.best_params_)
print("Best CV score (F1):", gs.best_score_)

# Get best model
best_rf_pipeline = gs.best_estimator_
# Evaluation helper
def evaluate_model(pipeline, X_test, y_test, name='Model'):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(f"--- {name} ---")
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1:', f1_score(y_test, y_pred))
    print('ROC AUC:', roc_auc_score(y_test, y_proba))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred))

# Evaluate on test data
evaluate_model(best_rf_pipeline, X_test, y_test, name='RandomForest (GridSearch tuned)')

# Save best model
model_path = R"C:\Users\acer\Desktop\Anaconda\liver_model_best.pkl"
joblib.dump(best_rf_pipeline, model_path)
print("Saved tuned model to:", model_path)
# Example prediction function
def predict_sample(sample_dict, model_path="C:\\Users\\acer\\Desktop\\Anaconda\\liver_model_best.pkl"):
    # Import necessary libraries if not already imported
    import joblib
    import pandas as pd
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return {'error': f"Model file not found at {model_path}. Please provide the correct path to the model file."}
    
    if 'Gender' in sample_dict and isinstance(sample_dict['Gender'], str):
        sample_dict = sample_dict.copy()
        sample_dict['Gender'] = 1 if sample_dict['Gender'].upper() == 'M' else 0
    sample_df = pd.DataFrame([sample_dict])
    
    try:
        proba = model.predict_proba(sample_df)[:, 1][0]
        pred = int(model.predict(sample_df)[0])
        return {'prediction': pred, 'probability_of_liver_disease': float(proba)}
    except Exception as e:
        return {'error': f"Error during prediction: {str(e)}"}

# Example
sample = {'Age':58,'Gender':'M','Total_Bilirubin':3.2,'Direct_Bilirubin':1.4,'Alkaline_Phosphotase':160,'ALT':120,'AST':110,'Total_Proteins':5.8,'Albumin':3.1,'A_G_Ratio':0.8}
print('Example prediction:', predict_sample(sample, model_path='C:\\Users\\acer\\Desktop\\Anaconda\\liver_model_best.pkl'))  # Update with your actual model path
```

### Output:
<img width="1253" height="318" alt="image" src="https://github.com/user-attachments/assets/cdb228ed-768e-43f1-9324-7301ed3f0882" />

<img width="1286" height="122" alt="image" src="https://github.com/user-attachments/assets/0d6b77c6-c897-471c-a40c-833e98b4e0a0" />

<img width="1289" height="442" alt="image" src="https://github.com/user-attachments/assets/e427cb51-e05c-4f11-bac1-22702317e872" />


### Result:
Thus the system was trained successfully and the prediction was carried out.
