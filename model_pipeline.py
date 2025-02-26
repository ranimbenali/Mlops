import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


def prepare_data(file_path="merged_churn.csv", test_size=0.2, random_state=42):
    """
    Charge et prétraite les données à partir d'un seul fichier.
    """
    data = pd.read_csv(file_path)

    # Identifier les colonnes catégoriques
    categorical_cols = data.select_dtypes(include=["object"]).columns

    # Appliquer Label Encoding sur les colonnes catégoriques
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraîne un modèle Random Forest.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}\n")
    print(report)
    return accuracy, report


def save_model(model, filename="random_forest_model.pkl"):
    """
    Sauvegarde le modèle entraîné.
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")


def load_model(filename="random_forest_model.pkl"):
    """
    Charge un modèle sauvegardé.
    """
    return joblib.load(filename)
