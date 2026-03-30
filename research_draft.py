import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
stats_df = pd.read_csv('PlayerStatistics.csv', low_memory=False)
players_df = pd.read_csv('Players.csv')
df = pd.merge(stats_df, players_df, on=['personId', 'firstName', 'lastName'], how='left')
df['fullName'] = df['firstName'].astype(str).str.strip() + ' ' + df['lastName'].astype(str).str.strip()
real_all_stars = [
    'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
    'Luka Doncic', 'Nikola Jokic', 'Joel Embiid', 'Jayson Tatum',
    'Devin Booker', 'Damian Lillard', 'Kyrie Irving', 'James Harden',
    'Kawhi Leonard', 'Anthony Davis', 'Paul George', 'Jimmy Butler',
    'Kobe Bryant', 'Michael Jordan', 'Shaquille O\'Neal', 'Tim Duncan',
    'Dwyane Wade', 'Dirk Nowitzki', 'Kevin Garnett', 'Allen Iverson',
    'Steve Nash', 'Jason Kidd', 'Chris Paul', 'Carmelo Anthony',
    'Russell Westbrook', 'Klay Thompson', 'Draymond Green', 'DeMar DeRozan',
    'Bradley Beal', 'Donovan Mitchell', 'Rudy Gobert', 'Karl-Anthony Towns',
    'Zion Williamson', 'Ja Morant', 'Trae Young', 'Shai Gilgeous-Alexander',
    'Anthony Edwards', 'Tyrese Haliburton', 'Jalen Brunson', 'Bam Adebayo'
]
df['is_all_star'] = (
    df['fullName'].isin(real_all_stars) &
    (df['points'] >= 15)).astype(int)
bool_columns = ['guard', 'forward', 'center']
for col in bool_columns:
    if col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0})
        elif df[col].dtype == bool:
            df[col] = df[col].astype(int)

features = ['points', 'assists', 'blocks', 'steals', 'reboundsTotal', 'turnovers',
            'plusMinusPoints', 'heightInches', 'bodyWeightLbs', 'guard', 'forward', 'center']

features = [f for f in features if f in df.columns]

for f in features:
    df[f] = pd.to_numeric(df[f], errors='coerce')

X = df[features]
y = df['is_all_star']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, features)])

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=20,
        random_state=42
    ))
])
xgb_pipeline.fit(X_train, y_train)

importances = xgb_pipeline.named_steps['classifier'].feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df,
    hue='Feature',
    palette='viridis',
    legend=False
)
plt.title('Top Features for NBA All-Star Selection (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

y_pred = xgb_pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(xgb_pipeline, 'nba_xgboost_model.joblib')