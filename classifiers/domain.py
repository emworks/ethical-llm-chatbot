import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from utils.ollama import embed
from utils.translation.lang import t
from utils.debug import d

MODEL_PATH = Path("classifiers/domain.joblib")


class DomainClassifier:
    DOMAINS = ["rag", "recommend", "table", "cot"]

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        if not MODEL_PATH.exists():
            print(t("domain_not_trained"))
            self.train()
        self.clf, self.domains = joblib.load(MODEL_PATH)

    def predict(self, text: str) -> str | None:
        try:
            emb = np.array(embed(text)).reshape(1, -1)
            probs = self.clf.predict_proba(emb)[0]

            for cls, p in zip(self.clf.classes_, probs):
                d(f"{cls}: {p:.3f}")
            
            best_idx = probs.argmax()
            if probs[best_idx] < self.threshold:
                return None
            return self.clf.classes_[best_idx]
        except Exception as e:
            print(t("error"), e)
            return None

    def train(self):
        X = [
            # RAG
            "Найди информацию в книге",
            "Ответь на вопрос по документу",
            "Find information in the text",
            "Beantworte eine Frage zum Dokument",
            "Trouve des informations dans le livre",
            # RECOMMEND
            "Порекомендуй книгу",
            "Дай рекомендацию на основе текста",
            "Recommend something based on the document",
            "Gib eine Empfehlung",
            "Recommande quelque chose basé sur le document",
            # TABLE
            "Сделай таблицу из текста",
            "Выведи таблицу объектов",
            "Create a table from the document",
            "Erstelle eine Tabelle aus dem Text",
            "Crée un tableau à partir du texte",
            # COT
            "Реши задачу пошагово",
            "Объясни решение с шагами",
            "Solve the problem step by step",
            "Löse die Aufgabe Schritt für Schritt",
            "Résous le problème étape par étape",
        ]
        y = (
            ["rag"] * 5
            + ["recommend"] * 5
            + ["table"] * 5
            + ["cot"] * 5
        )

        print(t("domain_training"))

        X_emb = np.array([embed(t) for t in X])

        clf = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_emb, y)

        joblib.dump((clf, self.DOMAINS), MODEL_PATH)

        print(t("domain_trained"))
