import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils import shuffle


data = pd.read_csv('Churn.csv')
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data['Tenure'] = data['Tenure'].fillna(data['Tenure'].median())

data_ohe = pd.get_dummies(data, drop_first=True)

target = data_ohe['Exited']
features = data_ohe.drop('Exited', axis=1)

features_train, features_temp, target_train, target_temp = train_test_split(
    features, target, test_size=0.4, random_state=12345)


features_valid, features_test, target_valid, target_test = train_test_split(
    features_temp, target_temp, test_size=0.5, random_state=12345)

numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()

scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])

model_base = RandomForestClassifier(random_state=12345)
model_base.fit(features_train, target_train)

predicted_valid = model_base.predict(features_valid)
probabilities_valid = model_base.predict_proba(features_valid)[:, 1]

print("--- Modelo Desbalanceado ---")
print("F1-Score:", f1_score(target_valid, predicted_valid))
print("AUC-ROC:", roc_auc_score(target_valid, probabilities_valid))

best_f1_weight = 0
best_depth_weight = 0

for depth in range(1, 16):
    model = RandomForestClassifier(random_state=12345, max_depth=depth, class_weight='balanced')
    model.fit(features_train, target_train)

    preds = model.predict(features_valid)
    f1 = f1_score(target_valid, preds)

    if f1 > best_f1_weight:
        best_f1_weight = f1
        best_depth_weight = depth

print("Mejor F1 con Pesos Balanceados:", best_f1_weight, "a profundidad", best_depth_weight)


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)

    return features_upsampled, target_upsampled



features_upsampled, target_upsampled = upsample(features_train, target_train, 4)

best_f1_up = 0
best_depth_up = 0
best_model_up = None

for depth in range(1, 16):

    model = RandomForestClassifier(random_state=12345, max_depth=depth)
    model.fit(features_upsampled, target_upsampled)

    preds = model.predict(features_valid)
    f1 = f1_score(target_valid, preds)

    if f1 > best_f1_up:
        best_f1_up = f1
        best_depth_up = depth
        best_model_up = model


probabilities_valid_up = best_model_up.predict_proba(features_valid)[:, 1]
auc_roc_up = roc_auc_score(target_valid, probabilities_valid_up)

print("--- Modelo con Sobremuestreo (Upsampling) ---")
print("Mejor F1-Score:", best_f1_up, "a profundidad", best_depth_up)
print("AUC-ROC:", auc_roc_up)

final_preds = best_model_up.predict(features_test)
final_probabilities = best_model_up.predict_proba(features_test)[:, 1]

test_f1 = f1_score(target_test, final_preds)
test_auc_roc = roc_auc_score(target_test, final_probabilities)
print(f"F1-Score Definitivo: {test_f1:.4f}")
print(f"AUC-ROC Definitivo:  {test_auc_roc:.4f}")