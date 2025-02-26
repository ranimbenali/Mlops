import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    args = parser.parse_args()

    file_path = "merged_churn.csv"

    if args.prepare:
        print("Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(file_path)
        print("Données préparées avec succès.")

    if args.train:
        print("Entraînement du modèle...")
        X_train, X_test, y_train, y_test = prepare_data(file_path)
        model = train_model(X_train, y_train)
        save_model(model)
        print("Modèle entraîné et sauvegardé.")

    if args.evaluate:
        print("Évaluation du modèle...")
        X_train, X_test, y_train, y_test = prepare_data(file_path)
        model = load_model()
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
