import pandas as pd

df = pd.read_csv("data/SPAS_with_months.csv")

# 1. Remove rows where Area = 0
df = df[df['Area'] > 0]

# 2. Remove rows where Production = 0
df = df[df['Production'] > 0]

df['AP Ratio'] = df['AP Ratio'].fillna(df['AP Ratio'].median())
df['Season'] = df['Season'].fillna(df['Season'].mode()[0])

print("After cleaning:", df.shape)

# create yield
df['Yield'] = df['Production'] / df['Area']

print("\n===== YIELD STATS =====")
print(df['Yield'].describe())

# remove extreme outliers (very high yield)
df = df[df['Yield'] < 50]

print("\nAfter removing outliers:", df.shape)
print(df['Yield'].describe())

# temperature variation
df['Temp_Range'] = df['Max Temp'] - df['Min Temp']

# humidity variation
df['Humidity_Range'] = df['Max Relative Humidity'] - df['Min Relative Humidity']

# climate interaction
df['Climate_Index'] = df['Rainfall_mm'] * df['Avg Temp']

# drop leakage + unnecessary columns
df = df.drop(columns=['Production', 'Transplant', 'Growth', 'Harvest'])

df = pd.get_dummies(df, columns=['District', 'Season', 'Crop Name'], drop_first=True)

X = df.drop(columns=['Yield'])
y = df['Yield']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

#RandomForest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n===== RANDOM FOREST =====")
print("MAE:", mean_absolute_error(y_test, pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))
print("R2:", r2_score(y_test, pred_rf))

#RIDGE regression
from sklearn.linear_model import Ridge

lr = Ridge()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("\n===== RIDGE REGRESSION =====")
print("MAE:", mean_absolute_error(y_test, pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("R2:", r2_score(y_test, pred_lr))

#Gradiant boosting
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

pred_gbr = gbr.predict(X_test)

print("\n===== GRADIENT BOOSTING =====")
print("MAE:", mean_absolute_error(y_test, pred_gbr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_gbr)))
print("R2:", r2_score(y_test, pred_gbr))

#1st stacking model
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor()),
    ('lr', Ridge())
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)

stack_model.fit(X_train, y_train)

pred_stack = stack_model.predict(X_test)

print("\n===== STACKING MODEL =====")
print("MAE:", mean_absolute_error(y_test, pred_stack))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_stack)))
print("R2:", r2_score(y_test, pred_stack))

#2nd stacking model
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor())
]

stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)

stack_model.fit(X_train, y_train)

pred_stack = stack_model.predict(X_test)

print("\n===== 2nd STACKING =====")
print("MAE:", mean_absolute_error(y_test, pred_stack))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_stack)))
print("R2:", r2_score(y_test, pred_stack))

#ExtraTress Regressor
from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor(n_estimators=100, random_state=42)
etr.fit(X_train, y_train)

pred_etr = etr.predict(X_test)

print("\n===== EXTRA TREES =====")
print("MAE:", mean_absolute_error(y_test, pred_etr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_etr)))
print("R2:", r2_score(y_test, pred_etr))

#KNN
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)

print("\n===== KNN =====")
print("MAE:", mean_absolute_error(y_test, pred_knn))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_knn)))
print("R2:", r2_score(y_test, pred_knn))

#combining diverse models. 
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor()),
    ('etr', ExtraTreesRegressor(n_estimators=100, random_state=42))
]

stack_model_new = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)

stack_model_new.fit(X_train, y_train)

pred_stack_new = stack_model_new.predict(X_test)

print("\n===== ADVANCED STACKING =====")
print("MAE:", mean_absolute_error(y_test, pred_stack_new))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_stack_new)))
print("R2:", r2_score(y_test, pred_stack_new))

#COMBINING all models for better diversity
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor()),
    ('etr', ExtraTreesRegressor(n_estimators=100, random_state=42)),
    ('lr', Ridge())
]

stack_model_full = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)

stack_model_full.fit(X_train, y_train)

pred_stack_full = stack_model_full.predict(X_test)

print("\n===== FULL STACKING (ALL MODELS) =====")
print("MAE:", mean_absolute_error(y_test, pred_stack_full))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_stack_full)))
print("R2:", r2_score(y_test, pred_stack_full))

#xgboost 
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)

print("\n===== XGBOOST =====")
print("MAE:", mean_absolute_error(y_test, pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_xgb)))
print("R2:", r2_score(y_test, pred_xgb))

#taking best models.
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor()),
    ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42))
]

from sklearn.ensemble import StackingRegressor

stack_final = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)

stack_final.fit(X_train, y_train)

pred_stack_final = stack_final.predict(X_test)

print("\n===== FINAL STACKING =====")
print("MAE:", mean_absolute_error(y_test, pred_stack_final))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_stack_final)))
print("R2:", r2_score(y_test, pred_stack_final))

#Tuned xgboost
xgb_new = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_new.fit(X_train, y_train)

pred_xgb_new = xgb_new.predict(X_test)

print("\n===== TUNED XGBOOST =====")
print("MAE:", mean_absolute_error(y_test, pred_xgb_new))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_xgb_new)))
print("R2:", r2_score(y_test, pred_xgb_new))

# ===== FEATURE IMPORTANCE (XGBOOST) =====

import matplotlib.pyplot as plt
import pandas as pd

# get importance from tuned model
importances = xgb_new.feature_importances_

# get feature names
feature_names = X.columns

# create dataframe
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# sort features
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

# print top 10
print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(feat_imp.head(10))

# normalize importance (optional but good for report)
feat_imp['Importance'] = feat_imp['Importance'] / feat_imp['Importance'].sum()

# plot top 10 features
ax = feat_imp.head(10).plot(
    x='Feature',
    y='Importance',
    kind='barh',
    figsize=(10, 6)
)

ax.set_title("Top 10 Important Features (XGBoost)")
ax.set_xlabel("Normalized Importance")
ax.set_ylabel("Feature")
ax.invert_yaxis()

plt.savefig("feature_importance.png")
plt.close()

#final comnining
from sklearn.ensemble import StackingRegressor

estimators_final = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor()),
    ('xgb', XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
]

stack_best = StackingRegressor(
    estimators=estimators_final,
    final_estimator=Ridge()
)

stack_best.fit(X_train, y_train)

pred_stack_best = stack_best.predict(X_test)

print("\n===== FINAL BEST STACKING =====")
print("MAE:", mean_absolute_error(y_test, pred_stack_best))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_stack_best)))
print("R2:", r2_score(y_test, pred_stack_best))

from sklearn.inspection import permutation_importance

# calculate importance
perm_importance = permutation_importance(
    stack_best, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# create dataframe
feat_imp_stack = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
})

# sort
feat_imp_stack = feat_imp_stack.sort_values(by='Importance', ascending=False)

print("\n===== STACKING TOP FEATURES =====")
print(feat_imp_stack.head(10))

import matplotlib.pyplot as plt

ax = feat_imp_stack.head(10).plot(
    x='Feature',
    y='Importance',
    kind='barh',
    figsize=(10, 6)
)

ax.set_title("Top 10 Features (Stacking Model)")
ax.invert_yaxis()
plt.savefig("stacking_feature_importance.png")
plt.close()


# ===== ACTUAL VS PREDICTED =====
import matplotlib.pyplot as plt

plt.scatter(y_test, pred_stack_best)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
plt.savefig("actual_vs_predicted.png")
plt.close()


# ===== RESIDUAL PLOT =====
residuals = y_test - pred_stack_best

plt.scatter(pred_stack_best, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Yield")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png")
plt.close()

# ===== CROSS-VALIDATION (5-FOLD) =====
from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    'Tuned XGBoost': XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    'Stacking (Best)': StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gbr', GradientBoostingRegressor()),
            ('xgb', XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ))
        ],
        final_estimator=Ridge()
    )
}

for name, model in cv_models.items():
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
        n_jobs=-1
    )
    if 'XGBoost' in name:
        cv_xgb = scores
    else:
        cv_stack = scores
    print(f"\n===== 5-FOLD CV: {name} =====")
    print(f"MAE:  {-scores['test_neg_mean_absolute_error'].mean():.4f} ± {scores['test_neg_mean_absolute_error'].std():.4f}")
    rmse_cv = np.sqrt(-scores['test_neg_mean_squared_error'])
    print(f"RMSE: {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}")
    print(f"R²:   {scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")

# ===== FEATURE SUBSET ANALYSIS =====
top_features = feat_imp_stack.head(10)['Feature'].tolist()

for n_features in [5, 7, 10]:
    selected = top_features[:n_features]
    
    X_sub = X[selected]
    X_train_sub = X_train[selected]
    X_test_sub = X_test[selected]
    
    stack_sub = StackingRegressor(
        estimators=estimators_final,
        final_estimator=Ridge()
    )
    stack_sub.fit(X_train_sub, y_train)
    pred_sub = stack_sub.predict(X_test_sub)
    
    print(f"\n===== STACKING WITH TOP {n_features} FEATURES =====")
    print(f"Features: {', '.join(selected)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred_sub):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_sub)):.4f}")
    print(f"R²:   {r2_score(y_test, pred_sub):.4f}")
    
    cv_scores = cross_validate(
        stack_sub, X_sub, y, cv=cv,
        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
        n_jobs=-1
    )
    print(f"CV MAE:  {-cv_scores['test_neg_mean_absolute_error'].mean():.4f} ± {cv_scores['test_neg_mean_absolute_error'].std():.4f}")
    print(f"CV RMSE: {np.sqrt(-cv_scores['test_neg_mean_squared_error']).mean():.4f} ± {np.sqrt(-cv_scores['test_neg_mean_squared_error']).std():.4f}")
    print(f"CV R²:   {cv_scores['test_r2'].mean():.4f} ± {cv_scores['test_r2'].std():.4f}")

# ===== 1. EDA =====
import seaborn as sns

# reload original data for categorical analysis
df_orig = pd.read_csv("data/SPAS_with_months.csv")
df_orig = df_orig[df_orig['Area'] > 0]
df_orig = df_orig[df_orig['Production'] > 0]
df_orig['AP Ratio'] = df_orig['AP Ratio'].fillna(df_orig['AP Ratio'].median())
df_orig['Season'] = df_orig['Season'].fillna(df_orig['Season'].mode()[0])
df_orig['Yield'] = df_orig['Production'] / df_orig['Area']
df_orig = df_orig[df_orig['Yield'] < 50]

summary = df_orig.describe(include='all').transpose()
summary.to_csv("summary_statistics.csv")

top10_feats = feat_imp_stack.head(10)['Feature'].tolist()
corr_df = X[top10_feats].copy()
corr_df['Yield'] = y
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title("Correlation Heatmap of Top 10 Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

plt.figure(figsize=(10, 6))
df_orig.boxplot(column='Yield', by='Season')
plt.title("Yield Distribution by Season"); plt.suptitle("")
plt.ylabel("Yield (tons/ha)"); plt.tight_layout()
plt.savefig("yield_by_season.png"); plt.close()

top10_crops = df_orig.groupby('Crop Name')['Yield'].median().sort_values(ascending=False).head(10).index
plt.figure(figsize=(14, 6))
df_orig[df_orig['Crop Name'].isin(top10_crops)].boxplot(column='Yield', by='Crop Name')
plt.title("Yield Distribution by Crop (Top 10)"); plt.suptitle("")
plt.xticks(rotation=45); plt.ylabel("Yield (tons/ha)"); plt.tight_layout()
plt.savefig("yield_by_crop.png"); plt.close()

print("\n===== EDA COMPLETE (plots + summary_statistics.csv) =====")

# ===== 3. STATISTICAL SIGNIFICANCE =====
from scipy import stats

t_stat, p_val = stats.ttest_rel(cv_xgb['test_r2'], cv_stack['test_r2'])

print(f"\n===== STATISTICAL SIGNIFICANCE (Paired t-test) =====")
print(f"XGBoost R² per fold:  {cv_xgb['test_r2']}")
print(f"Stacking R² per fold: {cv_stack['test_r2']}")
print(f"t = {t_stat:.4f}, p = {p_val:.6f}")
print(f"Significant at 0.05 level: {'Yes' if p_val < 0.05 else 'No'}")

# ===== 4. ERROR ANALYSIS =====
residuals = y_test - pred_stack_best

for feat in ['AP Ratio', 'Area', 'Avg Temp', 'Avg Humidity', 'Rainfall_mm']:
    if feat in X_test.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(X_test[feat], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel(feat); plt.ylabel("Residuals")
        plt.title(f"Residuals vs {feat}")
        plt.tight_layout()
        plt.savefig(f"residuals_vs_{feat.replace(' ', '_')}.png")
        plt.close()

print("\n===== ERROR ANALYSIS PLOTS SAVED =====")

# ===== 9. EXPORT RESULTS =====
results_df = pd.DataFrame({
    'Model': [
        'Random Forest', 'Ridge Regression', 'Gradient Boosting',
        'Extra Trees', 'KNN', 'XGBoost', 'Tuned XGBoost',
        'Stacking (Best)'
    ],
    'MAE': [
        mean_absolute_error(y_test, pred_rf),
        mean_absolute_error(y_test, pred_lr),
        mean_absolute_error(y_test, pred_gbr),
        mean_absolute_error(y_test, pred_etr),
        mean_absolute_error(y_test, pred_knn),
        mean_absolute_error(y_test, pred_xgb),
        mean_absolute_error(y_test, pred_xgb_new),
        mean_absolute_error(y_test, pred_stack_best)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, pred_rf)),
        np.sqrt(mean_squared_error(y_test, pred_lr)),
        np.sqrt(mean_squared_error(y_test, pred_gbr)),
        np.sqrt(mean_squared_error(y_test, pred_etr)),
        np.sqrt(mean_squared_error(y_test, pred_knn)),
        np.sqrt(mean_squared_error(y_test, pred_xgb)),
        np.sqrt(mean_squared_error(y_test, pred_xgb_new)),
        np.sqrt(mean_squared_error(y_test, pred_stack_best))
    ],
    'R2': [
        r2_score(y_test, pred_rf),
        r2_score(y_test, pred_lr),
        r2_score(y_test, pred_gbr),
        r2_score(y_test, pred_etr),
        r2_score(y_test, pred_knn),
        r2_score(y_test, pred_xgb),
        r2_score(y_test, pred_xgb_new),
        r2_score(y_test, pred_stack_best)
    ]
})
results_df.to_csv("model_comparison.csv", index=False)

cv_results_df = pd.DataFrame({
    'Model': ['Tuned XGBoost', 'Stacking (Best)'],
    'CV_MAE': [
        -cv_xgb['test_neg_mean_absolute_error'].mean(),
        -cv_stack['test_neg_mean_absolute_error'].mean()
    ],
    'CV_MAE_std': [
        cv_xgb['test_neg_mean_absolute_error'].std(),
        cv_stack['test_neg_mean_absolute_error'].std()
    ],
    'CV_RMSE': [
        np.sqrt(-cv_xgb['test_neg_mean_squared_error']).mean(),
        np.sqrt(-cv_stack['test_neg_mean_squared_error']).mean()
    ],
    'CV_RMSE_std': [
        np.sqrt(-cv_xgb['test_neg_mean_squared_error']).std(),
        np.sqrt(-cv_stack['test_neg_mean_squared_error']).std()
    ],
    'CV_R2': [
        cv_xgb['test_r2'].mean(),
        cv_stack['test_r2'].mean()
    ],
    'CV_R2_std': [
        cv_xgb['test_r2'].std(),
        cv_stack['test_r2'].std()
    ]
})
cv_results_df.to_csv("cv_comparison.csv", index=False)

print("\n===== RESULTS EXPORTED (model_comparison.csv, cv_comparison.csv) =====")

# ====================================================================
# STRATIFIED CV BY CROP GROUP
# ====================================================================
print("\n\n===== STRATIFIED CV BY CROP GROUP =====")
from sklearn.model_selection import StratifiedKFold

df_cv = pd.read_csv("data/SPAS_with_months.csv")
df_cv['Crop Name'] = df_cv['Crop Name'].str.strip().str.title()
df_cv.loc[df_cv['Crop Name'] == 'Lady Finger', 'Crop Name'] = 'Lady Finger'
df_cv.loc[df_cv['Crop Name'].str.contains('Lady', case=False), 'Crop Name'] = 'Lady Finger'
df_cv.loc[df_cv['Crop Name'].str.contains('Onion', case=False), 'Crop Name'] = 'Onion'
df_cv = df_cv[df_cv['Area'] > 0]; df_cv = df_cv[df_cv['Production'] > 0]
df_cv['AP Ratio'] = df_cv['AP Ratio'].fillna(df_cv['AP Ratio'].median())
df_cv['Season'] = df_cv['Season'].fillna(df_cv['Season'].mode()[0])
df_cv['Yield'] = df_cv['Production'] / df_cv['Area']
df_cv = df_cv[df_cv['Yield'] < 50]
df_cv['Temp_Range'] = df_cv['Max Temp'] - df_cv['Min Temp']
df_cv['Humidity_Range'] = df_cv['Max Relative Humidity'] - df_cv['Min Relative Humidity']
df_cv['Climate_Index'] = df_cv['Rainfall_mm'] * df_cv['Avg Temp']
df_cv = df_cv.drop(columns=['Production', 'Transplant', 'Growth', 'Harvest'])

rice = ['Aman', 'Aus', 'Boro']
veg = ['Barbati', 'Beans', 'Cabbage', 'Carrot', 'Cauliflower', 'Chalkumra', 'Chili', 'Cucumber',
       'Danta', 'Danta Shak', 'Gourd', 'Jhinga', 'Kakrol', 'Karala', 'Kolmi Shak',
       'Lady Finger', 'Lal Shak', 'Laushak', 'Mukhi Kachu', 'Oal Kachu', 'Palong Shak',
       'Patal', 'Puishak', 'Pumpkin', 'Radish', 'Shalgom', 'Sweet Potato']
fruits = ['Amra', 'Banana', 'Betelnut', 'Black Berry', 'Boroi', 'Dalim', 'Date Palm',
          'Green Coconut', 'Green Palmyra', 'Green Papaya', 'Guava', 'Jack Fruit',
          'Jambura', 'Jamrul', 'Lemon', 'Malta', 'Mango', 'Palmyra Palm', 'Pineapple',
          'Ripe Papaya', 'Safeda', 'Taramind', 'Wood Apple']
pulses = ['Arhar', 'Cheena', 'Gram', 'Groundnut', 'Lentil', 'Mashkalai', 'Motor', 'Mug', 'Sesame']
spices = ['Garlic', 'Ginger', 'Onion', 'Rape & Mustard']
other =  ['Jute', 'Maize 1', 'Maize 2', 'Sugarcane', 'Tobacco', 'Wheat']

def get_group(crop):
    if crop in rice: return 0
    if crop in veg: return 1
    if crop in fruits: return 2
    if crop in pulses: return 3
    if crop in spices: return 4
    return 5

df_cv['Crop_Group'] = df_cv['Crop Name'].apply(get_group)
print(f"Group distribution:\n{df_cv['Crop_Group'].value_counts().sort_index()}")

df_cv = pd.get_dummies(df_cv, columns=['District', 'Season', 'Crop Name'], drop_first=True)
X_cv = df_cv.drop(columns=['Yield', 'Crop_Group']); y_cv = df_cv['Yield']
groups = df_cv['Crop_Group'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(skf.split(X_cv, groups))

xgb_cv_s = cross_validate(
    XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
    X_cv, y_cv, cv=cv_splits, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=-1
)
stack_cv_s = cross_validate(
    StackingRegressor(estimators=estimators_final, final_estimator=Ridge()),
    X_cv, y_cv, cv=cv_splits, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=-1
)

print(f"\n--- Stratified CV by Crop Group ---")
print(f"XGBoost  — MAE: {-xgb_cv_s['test_neg_mean_absolute_error'].mean():.4f} ± {xgb_cv_s['test_neg_mean_absolute_error'].std():.4f}, R²: {xgb_cv_s['test_r2'].mean():.4f} ± {xgb_cv_s['test_r2'].std():.4f}")
print(f"Stacking — MAE: {-stack_cv_s['test_neg_mean_absolute_error'].mean():.4f} ± {stack_cv_s['test_neg_mean_absolute_error'].std():.4f}, R²: {stack_cv_s['test_r2'].mean():.4f} ± {stack_cv_s['test_r2'].std():.4f}")
print(f"(Random CV: XGB R² 0.826 ± 0.06, Stack R² 0.754 ± 0.07)")

# ====================================================================
# GROUP K-FOLD: Hold out entire crop categories
# ====================================================================
print("\n\n===== GROUP K-FOLD (HOLD OUT CROP CATEGORIES) =====")
from sklearn.model_selection import GroupKFold

df_gk = pd.read_csv("data/SPAS_with_months.csv")
df_gk = df_gk[df_gk['Area'] > 0]; df_gk = df_gk[df_gk['Production'] > 0]
df_gk['AP Ratio'] = df_gk['AP Ratio'].fillna(df_gk['AP Ratio'].median())
df_gk['Season'] = df_gk['Season'].fillna(df_gk['Season'].mode()[0])
df_gk['Yield'] = df_gk['Production'] / df_gk['Area']
df_gk = df_gk[df_gk['Yield'] < 50]
df_gk['Temp_Range'] = df_gk['Max Temp'] - df_gk['Min Temp']
df_gk['Humidity_Range'] = df_gk['Max Relative Humidity'] - df_gk['Min Relative Humidity']
df_gk['Climate_Index'] = df_gk['Rainfall_mm'] * df_gk['Avg Temp']
df_gk = df_gk.drop(columns=['Production', 'Transplant', 'Growth', 'Harvest'])

cereals = ['Aman', 'Aus', 'Boro', 'Wheat', 'Maize 1', 'Maize 2']
vegetables = ['Cabbage', 'Carrot', 'Cauliflower', 'Cucumber', 'Pumpkin', 'Radish',
              'Sweet Potato', 'Barbati', 'Beans', 'Chalkumra', 'Danta', 'Danta Shak',
              'Gourd', 'Jhinga', 'Kakrol', 'Karala', 'Kolmi Shak', 'Lady Finger',
              'Lal Shak', 'Laushak', 'Mukhi Kachu', 'Oal Kachu', 'Palong Shak',
              'Patal', 'Puishak', 'Shalgom']
fruits = ['Jack Fruit', 'Mango', 'Ripe Papaya', 'Green Papaya', 'Banana',
          'Amra', 'Betelnut', 'Black Berry', 'Boroi', 'Dalim', 'Date Palm',
          'Green Coconut', 'Green Palmyra', 'Guava', 'Jambura', 'Jamrul',
          'Lemon', 'Malta', 'Palmyra Palm', 'Pineapple', 'Safeda', 'Taramind', 'Wood Apple']
spices = ['Chili', 'Ginger', 'Garlic', 'Onion', 'Rape & Mustard']

def get_gk_group(crop):
    if crop in cereals: return 'Cereals'
    if crop in vegetables: return 'Vegetables'
    if crop in fruits: return 'Fruits'
    if crop in spices: return 'Spices'
    return 'Other'

df_gk['Group'] = df_gk['Crop Name'].apply(get_gk_group)
print(f"Group sizes:\n{df_gk['Group'].value_counts()}")

df_gk = pd.get_dummies(df_gk, columns=['District', 'Season', 'Crop Name'], drop_first=True)
X_gk = df_gk.drop(columns=['Yield', 'Group']); y_gk = df_gk['Yield']
groups_gk = df_gk['Group'].values

gkf = GroupKFold(n_splits=4)
gk_splits = list(gkf.split(X_gk, y_gk, groups_gk))

print(f"  Fold splits (held-out category per fold):")
for i, (tr, te) in enumerate(gk_splits):
    held_out = df_gk['Group'].iloc[te].unique()
    print(f"    Fold {i+1}: train on 3 categories, test on '{held_out[0]}'")

xgb_gk = cross_validate(
    XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
    X_gk, y_gk, cv=gk_splits, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=-1
)
stack_gk = cross_validate(
    StackingRegressor(estimators=estimators_final, final_estimator=Ridge()),
    X_gk, y_gk, cv=gk_splits, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=-1
)

print(f"\n--- Group K-Fold (unseen crop categories) ---")
print(f"XGBoost  — MAE: {-xgb_gk['test_neg_mean_absolute_error'].mean():.4f} ± {xgb_gk['test_neg_mean_absolute_error'].std():.4f}, R²: {xgb_gk['test_r2'].mean():.4f} ± {xgb_gk['test_r2'].std():.4f}")
print(f"Stacking — MAE: {-stack_gk['test_neg_mean_absolute_error'].mean():.4f} ± {stack_gk['test_neg_mean_absolute_error'].std():.4f}, R²: {stack_gk['test_r2'].mean():.4f} ± {stack_gk['test_r2'].std():.4f}")

# ====================================================================
# EXPERIMENT 1: Remove Jack Fruit, Sugarcane, Papaya — retrain & compare
# ====================================================================
print("\n\n===== EXPERIMENT 1: REMOVING TOP CROPS =====")

df_e1 = pd.read_csv("data/SPAS_with_months.csv")
remove_crops = ['jack fruit', 'sugarcane', 'ripe papaya', 'green papaya']
df_e1 = df_e1[~df_e1['Crop Name'].str.lower().isin(remove_crops)]
print(f"Rows after removal: {len(df_e1)}")

df_e1 = df_e1[df_e1['Area'] > 0]; df_e1 = df_e1[df_e1['Production'] > 0]
df_e1['AP Ratio'] = df_e1['AP Ratio'].fillna(df_e1['AP Ratio'].median())
df_e1['Season'] = df_e1['Season'].fillna(df_e1['Season'].mode()[0])
df_e1['Yield'] = df_e1['Production'] / df_e1['Area']
df_e1 = df_e1[df_e1['Yield'] < 50]
df_e1['Temp_Range'] = df_e1['Max Temp'] - df_e1['Min Temp']
df_e1['Humidity_Range'] = df_e1['Max Relative Humidity'] - df_e1['Min Relative Humidity']
df_e1['Climate_Index'] = df_e1['Rainfall_mm'] * df_e1['Avg Temp']
df_e1 = df_e1.drop(columns=['Production', 'Transplant', 'Growth', 'Harvest'])
df_e1 = pd.get_dummies(df_e1, columns=['District', 'Season', 'Crop Name'], drop_first=True)

X_e1 = df_e1.drop(columns=['Yield']); y_e1 = df_e1['Yield']
Xtr_e1, Xte_e1, ytr_e1, yte_e1 = train_test_split(X_e1, y_e1, test_size=0.2, random_state=42)

xgb_e1 = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_e1.fit(Xtr_e1, ytr_e1); px_e1 = xgb_e1.predict(Xte_e1)

stack_e1 = StackingRegressor(estimators=estimators_final, final_estimator=Ridge())
stack_e1.fit(Xtr_e1, ytr_e1); ps_e1 = stack_e1.predict(Xte_e1)

print(f"XGBoost  — MAE: {mean_absolute_error(yte_e1, px_e1):.4f}, RMSE: {np.sqrt(mean_squared_error(yte_e1, px_e1)):.4f}, R2: {r2_score(yte_e1, px_e1):.4f}")
print(f"Stacking — MAE: {mean_absolute_error(yte_e1, ps_e1):.4f}, RMSE: {np.sqrt(mean_squared_error(yte_e1, ps_e1)):.4f}, R2: {r2_score(yte_e1, ps_e1):.4f}")
print(f"(Original — XGBoost R2: 0.836, Stacking R2: 0.824)")

# ====================================================================
# EXPERIMENT 2: Train on 71 crops, test on 3 unseen crops
# ====================================================================
print("\n\n===== EXPERIMENT 2: UNSEEN CROP GENERALIZATION =====")

df_e2 = pd.read_csv("data/SPAS_with_months.csv")
df_e2 = df_e2[df_e2['Area'] > 0]; df_e2 = df_e2[df_e2['Production'] > 0]
df_e2['AP Ratio'] = df_e2['AP Ratio'].fillna(df_e2['AP Ratio'].median())
df_e2['Season'] = df_e2['Season'].fillna(df_e2['Season'].mode()[0])
df_e2['Yield'] = df_e2['Production'] / df_e2['Area']
df_e2 = df_e2[df_e2['Yield'] < 50]
df_e2['Temp_Range'] = df_e2['Max Temp'] - df_e2['Min Temp']
df_e2['Humidity_Range'] = df_e2['Max Relative Humidity'] - df_e2['Min Relative Humidity']
df_e2['Climate_Index'] = df_e2['Rainfall_mm'] * df_e2['Avg Temp']
df_e2 = df_e2.drop(columns=['Production', 'Transplant', 'Growth', 'Harvest'])

# pick 3 unseen crops
np.random.seed(42)
crop_counts = df_e2['Crop Name'].value_counts()
test_crops = ['Jack Fruit', 'Sugarcane', 'Malta']
test_crops = [c for c in test_crops if c in crop_counts.index][:3]

train_mask = ~df_e2['Crop Name'].isin(test_crops)
test_mask = df_e2['Crop Name'].isin(test_crops)

df_train = df_e2[train_mask].copy()
df_test = df_e2[test_mask].copy()

print(f"Train crops: {df_train['Crop Name'].nunique()}, rows: {len(df_train)}")
print(f"Test  crops: {df_test['Crop Name'].unique()}, rows: {len(df_test)}")

# one-hot encode on train, align test
df_train = pd.get_dummies(df_train, columns=['District', 'Season', 'Crop Name'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['District', 'Season', 'Crop Name'], drop_first=True)
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

Xtr_e2 = df_train.drop(columns=['Yield']); ytr_e2 = df_train['Yield']
Xte_e2 = df_test.drop(columns=['Yield']); yte_e2 = df_test['Yield']

xgb_e2 = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_e2.fit(Xtr_e2, ytr_e2); px_e2 = xgb_e2.predict(Xte_e2)

stack_e2 = StackingRegressor(estimators=estimators_final, final_estimator=Ridge())
stack_e2.fit(Xtr_e2, ytr_e2); ps_e2 = stack_e2.predict(Xte_e2)

print(f"\nXGBoost  on unseen crops — MAE: {mean_absolute_error(yte_e2, px_e2):.4f}, RMSE: {np.sqrt(mean_squared_error(yte_e2, px_e2)):.4f}, R2: {r2_score(yte_e2, px_e2):.4f}")
print(f"Stacking on unseen crops — MAE: {mean_absolute_error(yte_e2, ps_e2):.4f}, RMSE: {np.sqrt(mean_squared_error(yte_e2, ps_e2)):.4f}, R2: {r2_score(yte_e2, ps_e2):.4f}")

# ====================================================================
# FINAL MODEL COMPARISON (Full Dataset)
# ====================================================================
print("\n\n===== FINAL MODEL COMPARISON (TRAINED ON FULL DATA) =====")

models_final = {
    'Tuned XGBoost (all 150 feat)': XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
    'Stacking Best (all 150 feat)': StackingRegressor(
        estimators=estimators_final, final_estimator=Ridge()),
    'Stacking Top 5 Feat': StackingRegressor(
        estimators=estimators_final, final_estimator=Ridge())
}

# subset to top 5 features
top5 = feat_imp_stack.head(5)['Feature'].tolist()
X_top5 = X[top5]

for name, model in models_final.items():
    X_use = X_top5 if 'Top 5' in name else X
    model.fit(X_use, y)
    preds = model.predict(X_use)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    
    print(f"\n{name}:")
    print(f"  In-sample  — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # 5-fold CV
    cv_out = cross_validate(model, X_use, y, cv=cv, scoring=['neg_mean_absolute_error', 'r2'], n_jobs=-1)
    print(f"  5-Fold CV  — MAE: {-cv_out['test_neg_mean_absolute_error'].mean():.4f} ± {cv_out['test_neg_mean_absolute_error'].std():.4f}, R²: {cv_out['test_r2'].mean():.4f} ± {cv_out['test_r2'].std():.4f}")

import os
for plot in ["feature_importance.png", "stacking_feature_importance.png", "actual_vs_predicted.png", "residual_plot.png"]:
    os.startfile(plot)

# ====================================================================
# SAVE FINAL MODELS (joblib pickle)
# ====================================================================
print("\n\n===== SAVING FINAL MODELS =====")
import joblib

xgb_final = models_final['Tuned XGBoost (all 150 feat)']
stack_final = models_final['Stacking Best (all 150 feat)']
stack_top5 = models_final['Stacking Top 5 Feat']

joblib.dump(xgb_final, 'xgb_final_model.pkl')
joblib.dump(stack_final, 'stack_final_model.pkl')
joblib.dump(stack_top5, 'stack_top5_model.pkl')
print("Saved: xgb_final_model.pkl, stack_final_model.pkl, stack_top5_model.pkl")

# ====================================================================
# SHAP ANALYSIS (XGBoost)
# ====================================================================
print("\n\n===== SHAP ANALYSIS (XGBoost) =====")
import shap

# Use a random subset for speed
np.random.seed(42)
sample_idx = np.random.choice(X.index, size=min(500, len(X)), replace=False)
X_samp = X.loc[sample_idx]

print(f"Computing SHAP values on {len(X_samp)} samples...")
explainer = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_samp)

# SHAP beeswarm summary
shap.summary_plot(shap_values, X_samp, show=False)
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_summary.png")

# SHAP bar plot (top features)
shap.summary_plot(shap_values, X_samp, plot_type="bar", show=False)
plt.savefig("shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_bar.png")

# ====================================================================
# PARTIAL DEPENDENCE PLOTS (Top 5 features from stacking)
# ====================================================================
print("\n\n===== PARTIAL DEPENDENCE PLOTS =====")
from sklearn.inspection import PartialDependenceDisplay

top5_feats = feat_imp_stack.head(5)['Feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.ravel()

# Convert to float to avoid int64 dtype issues with PDP
X_float = X.astype({col: 'float64' for col in top5_feats})

for i, feat in enumerate(top5_feats):
    ax = axes_flat[i]
    PartialDependenceDisplay.from_estimator(
        xgb_final, X_float, [feat], ax=ax, kind='average'
    )
    ax.set_title(f'PDP: {feat}')

axes_flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig("partial_dependence.png", dpi=150)
plt.close()
print("Saved partial_dependence.png")

for plot in ["shap_summary.png", "shap_bar.png", "partial_dependence.png"]:
    os.startfile(plot)

print("\nAll done!")
