import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils.translation.lang import t

MODEL_PATH = Path("classifiers/language.joblib")
LABELS = ["ru", "en", "de", "fr"]


class LanguageClassifier:
    def __init__(self):
        if not MODEL_PATH.exists():
            print(t("lang_clf_not_trained"))
            self.train()
        self.vectorizer, self.model = joblib.load(MODEL_PATH)

    def predict(self, text: str) -> str:
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def train(self):
        X = [
            # RU
            "Помоги найти информацию в книге",
            "Сделай краткий пересказ текста",
            "Какие темы поднимаются в документе",
            # EN
            "Give me a summary of the document",
            "Find relevant information in the text",
            "Create a table from this file",
            # DE
            "Gib mir eine Zusammenfassung des Textes",
            "Finde Informationen im Dokument",
            "Erstelle eine Tabelle aus dem Text",
            # FR
            "Donne-moi un résumé du document",
            "Trouve des informations pertinentes",
            "Crée un tableau à partir du texte",
        ]

        y = [
            "ru", "ru", "ru",
            "en", "en", "en",
            "de", "de", "de",
            "fr", "fr", "fr",
        ]

        print(t("lang_clf_training"))

        # используются символьные n-граммы, стандартный подход для language ID
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
        )

        Xv = vectorizer.fit_transform(X)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xv, y)

        joblib.dump((vectorizer, clf), MODEL_PATH)

        print(t("lang_clf_trained"))