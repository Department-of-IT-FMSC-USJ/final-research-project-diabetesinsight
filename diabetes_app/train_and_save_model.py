
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import pyreadstat
import pandas as pd
import pyreadstat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


print("--- Loading Core NHANES Datasets for df1 Creation ---")
try:
    nhanes_demo = pd.read_sas("datasets/DEMO_J.xpt")
    nhanes_bmx = pd.read_sas("datasets/BMX_J.xpt")
    nhanes_paq = pd.read_sas("datasets/PAQ_J.xpt")
    nhanes_slq = pd.read_sas("datasets/SLQ_J.xpt")
    nhanes_smq = pd.read_sas("datasets/SMQ_J.xpt")
    nhanes_alq = pd.read_sas("datasets/ALQ_J.xpt")
    nhanes_dpq = pd.read_sas("datasets/DPQ_J.xpt", format="xport")
    nhanes_diet = pd.read_sas("datasets/DR1TOT_J.xpt")
    nhanes_mcq = pd.read_sas("datasets/MCQ_J.xpt")
    nhanes_bpq = pd.read_sas("datasets/BPQ_J.xpt")
    nhanes_diq = pd.read_sas("datasets/DIQ_J.xpt")
    nhanes_kiq = pd.read_sas("datasets/KIQ_U_J.xpt")


    print(" All core datasets loaded successfully.")
except FileNotFoundError as e:
    print(f" Error: Dataset file not found.")
    print(f"Missing file: {e.filename}")
    exit()


print("\n--- Merging Core Datasets into df1 ---")

df1 = nhanes_demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']].copy()



def smart_merge(df_base, new_df, columns_to_merge):
    print(f" Merging columns: {columns_to_merge}")
    initial_shape = df_base.shape

    df_base = df_base.merge(new_df[['SEQN'] + columns_to_merge], on='SEQN', how='left')
    for col in columns_to_merge:
        print(f"  Missing % in '{col}': {df_base[col].isna().mean() * 100:.2f}%")
    print(f"  Shape after merge: {initial_shape} -> {df_base.shape}")
    return df_base




df1 = smart_merge(df1, nhanes_bmx, ['BMXWT', 'BMXHT', 'BMXBMI','BMXWAIST'])
df1 = smart_merge(df1, nhanes_paq, ['PAQ605'])
df1 = smart_merge(df1, nhanes_slq, ['SLD012'])
df1 = smart_merge(df1, nhanes_smq, ['SMQ020'])
df1 = smart_merge(df1, nhanes_alq, ['ALQ121'])
df1 = smart_merge(df1, nhanes_dpq, ['DPQ020'])
df1 = smart_merge(df1, nhanes_diet, ['DR1TKCAL', 'DR1TSUGR'])
df1 = smart_merge(df1, nhanes_mcq, ['MCQ300C','MCQ160G','MCQ160L'])
df1 = smart_merge(df1, nhanes_bpq, ['BPQ020','BPQ080'])
df1 = smart_merge(df1, nhanes_diq, ['DIQ010'])
df1 = smart_merge(df1, nhanes_kiq, ['KIQ022'])






print("\n--- Renaming Columns ---")
df1.rename(columns={
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Sex',
    'BMXBMI': 'BMI_Original',
    'PAQ605': 'PhysicalActivity',
    'SLD012': 'SleepHours',
    'SMQ020': 'Smoker',
    'ALQ121': 'AlcoholFrequency',
    'DPQ020': 'DepressionLevel',
    'DR1TKCAL': 'Calories',
    'DR1TSUGR': 'Sugar',
    'MCQ300C': 'FamilyHistoryDiabetes',
    'BPQ020': 'HadHighBP',
    'DIQ010': 'Diabetes',
    'BMXWT': 'Weight',
    'BMXHT': 'Height',
    'BMXWAIST': 'WaistCircumference',
    'BPQ080': 'HadHighCholesterol',
    'MCQ160G': 'Numbness',
    'MCQ160L': 'FootProblems',
    'KIQ022': 'KidneyProblems'


}, inplace=True)
print(" Columns renamed.")

print("\n--- Initial Data Cleaning ---")

print("Replacing common invalid codes (7, 9, 77, 99) with NaN...")
df1.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan}, inplace=True)
print(" Invalid codes replaced with NaN.")



print(df1.isnull().sum())
print(df1.shape)
print(df1['Diabetes'].value_counts())

df1.drop(columns = ['AlcoholFrequency','SleepHours'],inplace = True)

if 'DepressionLevel' in df1.columns:
    df1.drop(columns=['DepressionLevel'], inplace=True)
    print(" Dropped 'DepressionLevel' column.")

print(df1.shape)
print(df1.isnull().sum())
for col in df1.columns:
    print(f'{col} {df1[col].unique()}')

print(df1.head())


df1 = df1[df1['Diabetes'].notna()]
critical_cols = ['Age', 'Weight', 'Height', 'FamilyHistoryDiabetes']
df1 = df1.dropna(subset=critical_cols)





print(df1.isnull().sum())

df1.dropna(inplace=True)
print(df1.isnull().sum())
print(f"Final shape after all drops: {df1.shape}")


for column in df1.columns:
    print(f"\nColumn: '{column}'")
    print(f"  Number of unique values: {df1[column].nunique()}")
    print(f"  Unique values: {df1[column].unique()}")




binary_cols_to_convert = [
    'Smoker', 'AlcoholFrequency', 'FamilyHistoryDiabetes', 'HadHighBP', 'Sex','PhysicalActivity',
    'Numbness','FootProblems','HadHighCholesterol','KidneyProblems'
]
print("Converting '2.0' to '0.0' for binary features where '2' indicates 'No'...")
actual_binary_cols = [col for col in binary_cols_to_convert if col in df1.columns]
df1[actual_binary_cols] = df1[actual_binary_cols].replace({2.0: 0.0})
print(" Binary features converted.")


print("Processing 'Diabetes' target variable (2.0->0.0, 3.0->1.0)...")
df1['Diabetes'] = df1['Diabetes'].replace({3.0: 1.0, 2.0: 0.0})
print(f" Diabetes target variable distribution:\n{df1['Diabetes'].value_counts()}")


print("\n--- Deriving BMI from Weight and Height ---")

df1['Weight'] = pd.to_numeric(df1['Weight'], errors='coerce')
df1['Height'] = pd.to_numeric(df1['Height'], errors='coerce')


df1['BMI'] = df1['Weight'] / ((df1['Height'] / 100)**2)
print(" BMI derived from Weight and Height.")


if 'BMI_Original' in df1.columns:
    df1.drop(columns=['BMI_Original'], inplace=True)
    print(" Dropped original 'BMI_Original' column.")

#
print("\n--- Calculating BMR, Activity Factor, and EER ---")


def calculate_bmr(gender_code, weight, height, age):
    """Calculates Basal Metabolic Rate (BMR) using Mifflin-St Jeor equation."""

    if gender_code == 1:
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender_code == 0:
        return (10 * weight) + (6.25 * height) - (5 * age) - 161
    return np.nan

def get_activity_factor(physical_activity):
    return 1.55 if physical_activity == 1 else 1.2


for col in ['Sex', 'Weight', 'Height', 'Age', 'PhysicalActivity']:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')

df1['BMR'] = df1.apply(
    lambda row: calculate_bmr(row['Sex'], row['Weight'], row['Height'], row['Age']),
    axis=1
)
df1['ActivityFactor'] = df1['PhysicalActivity'].apply(get_activity_factor)
df1['EER'] = df1['BMR'] * df1['ActivityFactor']
print("✅ BMR, ActivityFactor, EER calculated.")


print("\n--- Categorizing Calories and Sugar ---")

def categorize_calories_eer(row):
    total_calories_reported = row['Calories']
    eer = row['EER']

    if pd.isna(eer) or eer <= 0 or pd.isna(total_calories_reported):
        return np.nan

    if total_calories_reported < (0.8 * eer):
        return 'Low'
    elif (0.8 * eer) <= total_calories_reported <= (1.2 * eer):
        return 'Medium'
    else:
        return 'High'

df1['Calories_Category'] = df1.apply(categorize_calories_eer, axis=1)


def categorize_sugar_df(row):
    total_sugar_grams = row['Sugar']
    total_calories_reported = row['Calories']

    if pd.isna(total_sugar_grams) or pd.isna(total_calories_reported) or total_calories_reported <= 0:
        return np.nan

    calories_from_sugar = total_sugar_grams * 4
    percent_calories_from_sugar = (calories_from_sugar / total_calories_reported) * 100

    if percent_calories_from_sugar < 5:
        return 'Low'
    elif 5 <= percent_calories_from_sugar < 10:
        return 'Medium'
    else:
        return 'High'

df1['Sugar_Category'] = df1.apply(categorize_sugar_df, axis=1)
print(" Calories and Sugar categorized.")


print("Applying one-hot encoding for Calories_Category and Sugar_Category...")
df1 = pd.get_dummies(df1, columns=['Calories_Category', 'Sugar_Category'], drop_first=False)


bool_cols = df1.select_dtypes(include='bool').columns
if not bool_cols.empty:
    df1[bool_cols] = df1[bool_cols].astype(int)
    print(" Boolean one-hot encoded columns converted to int (0/1).")
else:
    print("No boolean columns found for conversion.")


print(df1.shape)
print(df1.isna().sum())
print(df1.head())


print("\n--- Final Cleaning of df1 ---")
initial_rows = df1.shape[0]
df1.dropna(inplace=True)
rows_dropped = initial_rows - df1.shape[0]
print(f"✅ Dropped {rows_dropped} rows with remaining NaN values after all feature engineering.")
print(f"Final shape of df1: {df1.shape}")
print(f"Total missing values in final df1: {df1.isnull().sum().sum()}")


print("\n--- Preview of the created df1 DataFrame ---")
print(df1.head())
print("\nColumns in final df1:")
print(df1.columns.tolist())



print(df1.isnull().sum())
print(df1.shape)
print(df1['Diabetes'].value_counts())
print(df1.head())

import pandas as pd



def detect_outliers(df, features):
    outlier_summary = []

    for col in features:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        num_outliers = outliers.shape[0]


        outlier_summary.append({
            'Feature': col,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Num Outliers': num_outliers,
            'Percentage': (num_outliers / df.shape[0]) * 100
        })


    outlier_df = pd.DataFrame(outlier_summary)

    return outlier_df



continuous_columns = [
    'Age', 'BMI', 'Weight', 'Height', 'WaistCircumference', 'Calories', 'Sugar', 'Numbness', 'FootProblems',
    'KidneyProblems', 'BMR', 'ActivityFactor', 'EER'
]


outlier_summary = detect_outliers(df1, continuous_columns)


print("\nOutlier Summary:")
print(outlier_summary)
print(df1['Diabetes'].value_counts())
print(df1.shape)


feature_columns = [
    'Age', 'Sex', 'BMI', 'PhysicalActivity', 'Smoker', 'FamilyHistoryDiabetes', 'HadHighBP',
     'Numbness', 'FootProblems', 'KidneyProblems','WaistCircumference',
    'Calories', 'Sugar','BMR','EER'
]
for column in feature_columns:
    print(f"\nColumn: '{column}'")
    print(f"  Number of unique values: {df1[column].nunique()}")
    print(f"  Unique values: {df1[column].unique()}")

X = df1[feature_columns]


y = df1['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(X_test.columns)


# # ================= CatBoost Model ==================
# catboost_base = CatBoostClassifier(
#     loss_function='Logloss',
#     eval_metric='F1',
#     random_seed=42,
#     verbose=0
# )
#
# catboost_param_grid = {
#     'iterations': [100, 200],
#     'depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1],
#     'l2_leaf_reg': [1, 3, 5]
# }
#
# catboost_grid = GridSearchCV(
#     estimator=catboost_base,
#     param_grid=catboost_param_grid,
#     scoring='f1_macro',
#     cv=5,
#     n_jobs=-1,
#     verbose=1
# )
#
# catboost_grid.fit(X_train_resampled, y_train_resampled)
# catboost_best = catboost_grid.best_estimator_
#
# y_pred_catboost = catboost_best.predict(X_test)
# print("\nCatBoost Classification Report:")
# print(classification_report(y_test, y_pred_catboost))
#
# # ================= Random Forest Model ==================
# rf_base = RandomForestClassifier(random_state=42)
#
# rf_param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }
#
# rf_grid = GridSearchCV(
#     estimator=rf_base,
#     param_grid=rf_param_grid,
#     scoring='f1_macro',
#     cv=5,
#     n_jobs=-1,
#     verbose=1
# )
#
# rf_grid.fit(X_train_resampled, y_train_resampled)
# rf_best = rf_grid.best_estimator_
#
# y_pred_rf = rf_best.predict(X_test)
# print("\nRandom Forest Classification Report:")
# print(classification_report(y_test, y_pred_rf))
#
# # ================= XGBoost Model ==================
# xgb_base = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='logloss',
#     verbosity=0,
#     random_state=42
# )
#
# xgb_param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1],
#     'reg_lambda': [1, 3]
# }
#
# xgb_grid = GridSearchCV(
#     estimator=xgb_base,
#     param_grid=xgb_param_grid,
#     scoring='f1_macro',
#     cv=5,
#     n_jobs=-1,
#     verbose=1
# )
#
# xgb_grid.fit(X_train_resampled, y_train_resampled)
# xgb_best = xgb_grid.best_estimator_
#
# y_pred_xgb = xgb_best.predict(X_test)
# print("\nXGBoost Classification Report:")
# print(classification_report(y_test, y_pred_xgb))


from catboost import CatBoostClassifier


base_model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='F1',
    random_seed=42,
    verbose=0
)


param_grid = {
    'iterations': [100, 200],
    'depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'l2_leaf_reg': [1, 3, 5]
}


grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_resampled, y_train_resampled)
catboost_model = grid_search.best_estimator_

y_pred = catboost_model.predict(X_test)


print(f"\nCATBoost Classification Report:")
print(classification_report(y_test, y_pred))


explainer = shap.Explainer(catboost_model, X_train_resampled)
shap_values = explainer(X_test)


index = 48


pred = catboost_model.predict([X_test.iloc[index]])[0]
prob = catboost_model.predict_proba([X_test.iloc[index]])[0]


prediction_label = "Diabetic" if pred == 1 else "Not Diabetic"
prediction_prob = prob[int(pred)] * 100


shap_values_instance = shap.Explanation(
    values=shap_values[index].values,
    base_values=explainer.expected_value,
    data=X_test.iloc[index].values,
    feature_names=X_test.columns
)

fig, ax = plt.subplots(figsize=(16, len(X_test.columns) * 0.4))  # Adjust the height based on number of features
shap.plots.waterfall(shap_values_instance, max_display=len(X_test.columns), show=False)

plt.subplots_adjust(left=0.4, top=0.9)
plt.suptitle(f"Prediction: {prediction_label} ({prediction_prob:.1f}%)", fontsize=16, y=0.98)

plt.show()
# print(merged_df.shape)


import seaborn as sns
shap_matrix = np.abs(shap_values.values)

mean_shap = np.mean(shap_matrix, axis=0)


shap_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean |SHAP value|': mean_shap
}).sort_values(by='Mean |SHAP value|', ascending=False)


plt.figure(figsize=(10, 8))
sns.heatmap(shap_df.set_index('Feature').T, cmap="YlOrRd", annot=True, fmt=".3f", cbar=True)
plt.title('Mean SHAP Value per Feature (Feature Importance)')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


import skfuzzy as fuzz
from skfuzzy import control as ctrl


bmi = ctrl.Antecedent(np.arange(10, 50, 1), 'BMI')
sugar = ctrl.Antecedent(np.arange(0, 150, 1), 'Sugar')
waist = ctrl.Antecedent(np.arange(60, 140, 1), 'WaistCircumference')
activity = ctrl.Antecedent(np.arange(1, 3, 1), 'PhysicalActivity')
risk = ctrl.Consequent(np.arange(0, 100, 1), 'DiabetesRisk')


bmi['low'] = fuzz.trimf(bmi.universe, [10, 18.5, 25])
bmi['medium'] = fuzz.trimf(bmi.universe, [18.5, 25, 30])
bmi['high'] = fuzz.trimf(bmi.universe, [25, 35, 50])

sugar['low'] = fuzz.trimf(sugar.universe, [0, 20, 40])
sugar['medium'] = fuzz.trimf(sugar.universe, [30, 50, 70])
sugar['high'] = fuzz.trimf(sugar.universe, [60, 100, 150])

waist['low'] = fuzz.trimf(waist.universe, [60, 75, 90])
waist['medium'] = fuzz.trimf(waist.universe, [85, 100, 110])
waist['high'] = fuzz.trimf(waist.universe, [100, 115, 140])

activity['low'] = fuzz.trimf(activity.universe, [1.5, 2, 2])
activity['high'] = fuzz.trimf(activity.universe, [1, 1, 1.5])

risk['low'] = fuzz.trimf(risk.universe, [0, 20, 40])
risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
risk['high'] = fuzz.trimf(risk.universe, [60, 80, 100])


rules = [
    ctrl.Rule(bmi['high'] & sugar['high'] & waist['high'] & activity['low'], risk['high']),
    ctrl.Rule(bmi['medium'] & sugar['medium'] & waist['medium'] & activity['low'], risk['medium']),
    ctrl.Rule(bmi['low'] & sugar['low'] & waist['low'] & activity['high'], risk['low']),
    ctrl.Rule(bmi['high'] & activity['low'], risk['high']),
    ctrl.Rule(sugar['high'] & waist['medium'], risk['medium']),
    ctrl.Rule(bmi['medium'] & activity['high'], risk['low']),
]


risk_ctrl = ctrl.ControlSystem(rules)
fuzzy_simulator = ctrl.ControlSystemSimulation(risk_ctrl)

def fuzzy_logic(instance, top_shap_features, pred, WHO_thresholds):

    return " - Consider lifestyle adjustments to reduce BMI and sugar intake.\n"
def generate_detailed_report(instance_index, X_test, y_test, model, shap_values, fuzzy_logic, WHO_thresholds):
    instance = X_test.iloc[instance_index]
    pred = model.predict([instance])[0]
    prob = model.predict_proba([instance])[0][int(pred)]

    feature_recommendations = {
        'Age': "While age can't be changed, regular screenings and healthy lifestyle choices can delay diabetes progression.",
        'BMI': "Maintain a healthy BMI through balanced diet and regular physical activity.",
        'WaistCircumference': "A high waist circumference indicates abdominal fat; consider core-focused exercise and portion control.",
        'EER': "Ensure energy intake aligns with expenditure. Avoid frequent overeating.",
        'BMR': "Lower BMR indicates lower energy needs. Tailor calorie intake accordingly.",
        'Sex': "Biological sex influences diabetes risk; women may face additional hormonal risks. Monitor regularly.",
        'PhysicalActivity': "Increase activity to at least 150 minutes/week of moderate-intensity exercise.",
        'Smoker': "Quit smoking to improve insulin sensitivity and cardiovascular health.",
        'FamilyHistoryDiabetes': "With family history, routine checkups and preventive care are vital.",
        'Sugar': "Reduce daily sugar intake to below 10% of total calories, per WHO guidelines.",
        'Calories': "Avoid consistent overconsumption of calories; eat mindfully.",
        'Numbness': "Nerve-related symptoms like numbness may indicate diabetic neuropathy risk. Consult a physician.",
        'FootProblems': "Foot issues are common in diabetes. Regular checkups and proper footwear are recommended.",
        'KidneyProblems': "Monitor kidney function closely; diabetes is a major risk factor for kidney disease."
    }


    instance_shap_values = shap_values[instance_index].values
    shap_feature_pairs = list(zip(instance_shap_values, X_test.columns))


    risk_features = sorted(
        [(val, feat) for val, feat in shap_feature_pairs if val > 0],
        key=lambda x: abs(x[0]), reverse=True
    )[:5]


    recommendation_block = fuzzy_logic(instance, risk_features, pred, WHO_thresholds)


    bmi = instance['BMI']
    waist = instance['WaistCircumference']
    bmi_alert = "Obese" if bmi >= WHO_thresholds['bmi_obese'] else "Normal"
    waist_alert = "High" if waist >= WHO_thresholds['waist_high'] else "Normal"


    fuzzy_simulator.input['BMI'] = instance['BMI']
    fuzzy_simulator.input['Sugar'] = instance['Sugar']
    fuzzy_simulator.input['WaistCircumference'] = instance['WaistCircumference']
    fuzzy_simulator.input['PhysicalActivity'] = instance['PhysicalActivity']
    fuzzy_simulator.compute()
    fuzzy_risk_score = fuzzy_simulator.output['DiabetesRisk']
    fuzzy_risk_label = (
        'High' if fuzzy_risk_score >= 60 else
        'Medium' if fuzzy_risk_score >= 30 else
        'Low'
    )


    detailed_recommendations = []
    if len(risk_features) > 0:
        if pred == 1:
            detailed_recommendations.append(" - The following factors contributed to your high diabetes risk:")
        else:
            detailed_recommendations.append(" - Even though you're not diabetic, the following factors increased your risk:")
        for shap_val, feature in risk_features:
            if feature in feature_recommendations:
                detailed_recommendations.append(f"   ✓ {feature} (↑) ➜ {feature_recommendations[feature]}")
    else:
        detailed_recommendations.append(" - No significant high-risk features identified by SHAP.")


    report = f"""
===============================
  Patient Diabetes Risk Report
===============================

   Patient Info:
 - Age: {instance['Age']}
 - Sex: {'Male' if instance['Sex'] == 1 else 'Female'}
 - BMI: {bmi:.1f} ({bmi_alert})
 - Waist Circumference: {waist} cm ({waist_alert})
 - Physical Activity: {'Low' if instance['PhysicalActivity'] == 2 else 'High'}
 - Smoker: {'Yes' if instance['Smoker'] == 1 else 'No'}
 - Family History: {'Yes' if instance['FamilyHistoryDiabetes'] == 1 else 'No'}

   Derived Metrics:
 - BMR: {instance['BMR']:.1f} kcal/day
 - EER: {instance['EER']:.1f} kcal/day
 - Calories Reported: {instance['Calories']} kcal/day
 - Sugar: {instance['Sugar']} g/day

   Prediction (CatBoost + SHAP):
 - Prediction: {'Diabetic' if pred == 1 else 'Not Diabetic'} (Probability: {prob * 100:.1f}%)
 - SHAP Influences: {', '.join([f"{feature} ({value:.2f})" for value, feature in risk_features]) or 'None'}

   Fuzzy Diabetes Risk (FIS Inference):
 - Risk Score: {fuzzy_risk_score:.2f} / 100
 - Risk Category: {fuzzy_risk_label}

   SHAP-Based Detailed Recommendations:
{chr(10).join(detailed_recommendations)}

   Additional Recommendations:
{recommendation_block}

   Based on WHO, NHANES, SHAP & Fuzzy Inference
===============================
"""
    return report


WHO_thresholds = {
    'bmi_obese': 30,
    'waist_high': 102 if X_test.iloc[0]['Sex'] == 1 else 88  # 102cm men, 88cm women (WHO)
}


report = generate_detailed_report(
    instance_index=index,
    X_test=X_test,
    y_test=y_test,
    model=catboost_model,
    shap_values=shap_values,
    fuzzy_logic=fuzzy_logic,
    WHO_thresholds=WHO_thresholds
)
print(report)


import joblib
import os

print("\n--- Saving Model and Supporting Assets ---")


ASSETS_DIR = 'assets'
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)


model_path = os.path.join(ASSETS_DIR, 'catboost_model.pkl')
joblib.dump(catboost_model, model_path)
print(f"✅ Model saved to {model_path}")


explainer_path = os.path.join(ASSETS_DIR, 'shap_explainer.pkl')
joblib.dump(explainer, explainer_path)
print(f"✅ SHAP explainer saved to {explainer_path}")


features_path = os.path.join(ASSETS_DIR, 'feature_columns.pkl')
joblib.dump(feature_columns, features_path)
print(f"✅ Feature column list saved to {features_path}")