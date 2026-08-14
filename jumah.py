import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# ==========================================
# PART 1: THE STANDARD ML SUBMISSION (BLIND)
# ==========================================
print("Generating Part-1 Submission...")

# 1. Train Gradient Boosting on 100% of the raw training data
gb_model = GradientBoostingClassifier(random_state=42)
X_train_processed_full_p1 = preprocessor_p1.fit_transform(X_train_p1)
gb_model.fit(X_train_processed_full_p1, y_train_p1)

# 2. Process the Test Data and Predict
X_test_processed_p1 = preprocessor_p1.transform(test)
part1_predictions = gb_model.predict(X_test_processed_p1)

# 3. Save to CSV
submission_p1 = pd.DataFrame({
    'ID': test['ID'], # Change 'ID' if your test dataset uses a different identifier column name
    'Oil_Presence': part1_predictions
})
submission_p1.to_csv('submission_part1.csv', index=False)
print("-> 'submission_part1.csv' saved successfully.\n")


# ==========================================
# PART 2: THE PHYSICS-INFORMED SUBMISSION
# ==========================================
print("Generating Part-2 Submission...")

# 1. Prepare the Test Data (Apply Physics Rules)
test_p2 = test.copy()

# A. Shale Override
shale_missing_test = (test_p2['Rock_Type'] == 'Shale') & (test_p2['Permeability (mD)'].isnull())
test_p2.loc[shale_missing_test, 'Permeability (mD)'] = 1.0

# B. One-Hot Encoding (Ensure columns align perfectly with X_p2)
test_encoded = pd.get_dummies(test_p2, columns=['Rock_Type', 'Trap_Type'], dtype=float)
test_encoded = test_encoded.reindex(columns=X_p2.columns, fill_value=0) # Aligns columns, fills missing categories with 0

# C. KNN Imputation (Using the training data as the anchor)
imputer_final = KNNImputer(n_neighbors=5, weights='distance')
# We fit the imputer on our pristine X_p2 training data, then transform the test data
imputer_final.fit(X_p2)
test_encoded[X_p2.columns] = imputer_final.transform(test_encoded[X_p2.columns])

# D. Clamp Physical Boundaries
test_encoded['Porosity (%)'] = test_encoded['Porosity (%)'].clip(lower=0, upper=30)
test_encoded['Permeability (mD)'] = test_encoded['Permeability (mD)'].clip(lower=1, upper=1000)
if 'Proximity_to_Oil_Field (km)' in test_encoded.columns:
    test_encoded['Proximity_to_Oil_Field (km)'] = test_encoded['Proximity_to_Oil_Field (km)'].clip(lower=0, upper=5)

# 2. Scale the Data
scaler_final = StandardScaler()
X_train_p2_scaled_full = scaler_final.fit_transform(X_p2)
X_test_p2_scaled = scaler_final.transform(test_encoded)

# 3. Train SVM on 100% of the scaled training data and Predict
svm_model = SVC(class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train_p2_scaled_full, y_p2)
part2_predictions = svm_model.predict(X_test_p2_scaled)

# 4. Save to CSV
submission_p2 = pd.DataFrame({
    'ID': test['ID'],
    'Oil_Presence': part2_predictions
})
submission_p2.to_csv('submission_part2.csv', index=False)
print("-> 'submission_part2.csv' saved successfully.")
print("\nCongratulations! Both submission files are ready.")