MESSAGES = {
    "install_deps": {
        "ru": "Установите зависимости: pip install chromadb ollama nltk pydantic telebot",
        "en": "Install dependencies: pip install chromadb ollama nltk pydantic telebot",
        "de": "Installieren Sie die Abhängigkeiten: pip install chromadb ollama nltk pydantic telebot",
        "fr": "Installez les dépendances : pip install chromadb ollama nltk pydantic telebot",
    },
    "dir_not_found": {
        "ru": "Каталог {dir_path} не найден",
        "en": "Directory {dir_path} not found",
        "de": "Verzeichnis {dir_path} wurde nicht gefunden",
        "fr": "Le répertoire {dir_path} est introuvable",
    },
    "db_up_to_date": {
        "ru": "База {collection_name} актуальна.",
        "en": "Database {collection_name} is up to date.",
        "de": "Datenbank {collection_name} ist aktuell.",
        "fr": "La base de données {collection_name} est à jour.",
    },
    "db_building": {
        "ru": "Строим базу {collection_name}...",
        "en": "Building database {collection_name}...",
        "de": "Datenbank {collection_name} wird erstellt...",
        "fr": "Construction de la base de données {collection_name}...",
    },
    "db_built": {
        "ru": "База {collection_name} построена.",
        "en": "Database {collection_name} has been built.",
        "de": "Datenbank {collection_name} wurde erstellt.",
        "fr": "La base de données {collection_name} a été construite.",
    },
    "system_prompt_rag": {
        "ru": "Ты помощник, отвечающий строго по базе. Если ответа нет, скажи 'Информация не найдена в базе'.",
        "en": "You are an assistant that answers strictly based on the database. If no answer is found, say 'Information not found in the database'.",
        "de": "Du bist ein Assistent, der strikt auf Basis der Datenbank antwortet. Wenn keine Antwort gefunden wird, sage 'Information nicht in der Datenbank gefunden'.",
        "fr": "Tu es un assistant qui répond strictement à partir de la base de données. Si aucune information n'est trouvée, dis 'Information non trouvée dans la base de données'.",
    },
    "context_truncated": {
        "ru": "...(обрезано)",
        "en": "...(truncated)",
        "de": "...(gekürzt)",
        "fr": "...(tronqué)",
    },
    "rag_user_prompt": {
        "ru": "Контекст:\n{context}\nВопрос: {question}",
        "en": "Context:\n{context}\nQuestion: {question}",
        "de": "Kontext:\n{context}\nFrage: {question}",
        "fr": "Contexte :\n{context}\nQuestion : {question}",
    },
    "table_extract_system": {
        "ru": '''
            Извлеки из текста все достопримечательности/объекты и их категории (domain).
            Примеры категорий: архитектура, история, культура, религия, природа и т.д.
            У одного объекта может быть несколько категорий – перечисли все подходящие. 
            Верни строго JSON: {"table": [{"name": "...", "domain": "..."}, ...]}.
        ''',
        "en": '''
            Extract all landmarks/objects and their categories (domain) from the text.
            Example categories: architecture, history, culture, religion, nature, etc.
            An object may have multiple categories — list all applicable ones.
            Return strictly JSON: {"table": [{"name": "...", "domain": "..."}, ...]}.
        ''',
        "de": '''
            Extrahiere alle Sehenswürdigkeiten/Objekte und ihre Kategorien (Domain) aus dem Text.
            Beispielkategorien: Architektur, Geschichte, Kultur, Religion, Natur usw.
            Ein Objekt kann mehrere Kategorien haben — liste alle passenden auf.
            Gib ausschließlich JSON zurück: {"table": [{"name": "...", "domain": "..."}, ...]}.
        ''',
        "fr": '''
            Extrais du texte toutes les attractions/objets et leurs catégories (domaine).
            Exemples de catégories : architecture, histoire, culture, religion, nature, etc.
            Un objet peut appartenir à plusieurs catégories — liste-les toutes.
            Retourne strictement du JSON : {"table": [{"name": "...", "domain": "..."}, ...]}.
        ''',
    },
    "table_extract_error": {
        "ru": "Ошибка извлечения таблицы:",
        "en": "Table extraction error:",
        "de": "Fehler bei der Tabellenextraktion:",
        "fr": "Erreur lors de l'extraction du tableau :",
    },
    "table_md_header": {
        "ru": "| Достопримечательность | Домен |\n| :--- | :--- |\n",
        "en": "| Landmark | Domain |\n| :--- | :--- |\n",
        "de": "| Sehenswürdigkeit | Domäne |\n| :--- | :--- |\n",
        "fr": "| Attraction | Domaine |\n| :--- | :--- |\n",
    },
    "table_plain_header": {
        "ru": "Достопримечательность        | Домен\n",
        "en": "Landmark                     | Domain\n",
        "de": "Sehenswürdigkeit             | Domäne\n",
        "fr": "Attraction                   | Domaine\n",
    },
    "cot_example": {
        "ru": "Пример",
        "en": "Example",
        "de": "Beispiel",
        "fr": "Exemple",
    },
    "cot_system": {
        "ru": """
            Ты ассистент, который решает текстовые задачи пошагово.
            Каждое рассуждение оформляй как шаг, нумеруя их: Шаг 1., Шаг 2., ...
            В конце укажи правильный ответ.
            Используй приведенные примеры как руководство по стилю решения.
        """,
        "en": """
            You are an assistant that solves text-based problems step by step.
            Format each reasoning step as Step 1, Step 2, etc.
            Provide the final correct answer at the end.
            Use the given examples as guidance for the solution style.
        """,
        "de": """
            Du bist ein Assistent, der textbasierte Aufgaben schrittweise löst.
            Formatiere jeden Gedankenschritt als Schritt 1, Schritt 2 usw.
            Gib am Ende die richtige Antwort an.
            Nutze die Beispiele als stilistische Orientierung.
        """,
        "fr": """
            Tu es un assistant qui résout des tâches textuelles étape par étape.
            Formate chaque raisonnement comme Étape 1, Étape 2, etc.
            Indique la réponse correcte à la fin.
            Utilise les exemples fournis comme guide de style.
        """,
    },
    "cot_user": {
        "ru": "{few_shot_context}\nНовая задача: {task_text}",
        "en": "{few_shot_context}\nNew task: {task_text}",
        "de": "{few_shot_context}\nNeue Aufgabe: {task_text}",
        "fr": "{few_shot_context}\nNouvelle tâche: {task_text}",
    },
    "cot_error": {
        "ru": "Ошибка при решении задачи:",
        "en": "Error while solving the task:",
        "de": "Fehler bei der Aufgabenlösung:",
        "fr": "Erreur lors de la résolution de la tâche :",
    },
    "cot_fail": {
        "ru": "Не удалось решить задачу.",
        "en": "Failed to solve the task.",
        "de": "Aufgabe konnte nicht gelöst werden.",
        "fr": "Impossible de résoudre la tâche.",
    },
    "mode_1": {
        "ru": "1 – Вопрос по базе",
        "en": "1 – Question based on the database",
        "de": "1 – Frage zur Datenbank",
        "fr": "1 – Question sur la base de données",
    },
    "mode_2": {
        "ru": "2 – Рекомендация",
        "en": "2 – Recommendation",
        "de": "2 – Empfehlung",
        "fr": "2 – Recommandation",
    },
    "mode_3": {
        "ru": "3 – Таблица из текста",
        "en": "3 – Table from text",
        "de": "3 – Tabelle aus Text",
        "fr": "3 – Tableau à partir du texte",
    },
    "mode_4": {
        "ru": "4 – Пошаговое решение задачи",
        "en": "4 – Step-by-step problem solving",
        "de": "4 – Schrittweise Aufgabenlösung",
        "fr": "4 – Résolution étape par étape",
    },
    "mode_5": {
        "ru": "5 – Добавить текст в базу",
        "en": "5 – Add text to database",
        "de": "5 – Text zur Datenbank hinzufügen",
        "fr": "5 – Ajouter un texte à la base",
    },
    "mode_6": {
        "ru": "6 – Пересказ и тэги",
        "en": "6 – Summary & Tags",
        "de": "6 – Zusammenfassung & Tags",
        "fr": "6 – Résumé & Tags",
    },
    "mode_skip": {
        "ru": "Пропустить",
        "en": "Skip",
        "de": "Skip",
        "fr": "Skip",
    },
    "chat_started": {
        "ru": "Чат запущен. Для выхода: Ctrl-C или 'exit'.",
        "en": "Chat started. To exit: Ctrl-C or 'exit'.",
        "de": "Chat gestartet. Zum Beenden: Ctrl-C oder 'exit'.",
        "fr": "Chat démarré. Pour quitter: Ctrl-C ou 'exit'.",
    },
    "choose_mode": {
        "ru": "Выберите режим:\n",
        "en": "Choose a mode:\n",
        "de": "Wählen Sie einen Modus:\n",
        "fr": "Choisissez un mode:\n",
    },
    "query_choice": {
        "ru": "Выбор: ",
        "en": "Choice: ",
        "de": "Auswahl: ",
        "fr": "Choix: ",
    },
    "invalid_choice": {
        "ru": "Некорректный выбор. Попробуйте снова.",
        "en": "Invalid choice. Try again.",
        "de": "Ungültige Auswahl. Bitte erneut versuchen.",
        "fr": "Choix invalide. Réessayez.",
    },
    "query_prompt": {
        "ru": "Запрос: ",
        "en": "Query: ",
        "de": "Anfrage: ",
        "fr": "Requête: ",
    },
    "moral_block_bot": {
        "ru": "Бот: К сожалению, ваш запрос нарушает мои моральные принципы и не будет обработан.",
        "en": "Bot: Sorry, your request violates my moral principles and cannot be processed.",
        "de": "Bot: Leider verstößt Ihre Anfrage gegen meine moralischen Prinzipien.",
        "fr": "Bot: Désolé, votre demande enfreint mes principes moraux.",
    },
    "bot_prefix": {
        "ru": "Бот:",
        "en": "Bot:",
        "de": "Bot:",
        "fr": "Bot:",
    },
    "table_fail_bot": {
        "ru": "Бот: Не удалось извлечь таблицу.",
        "en": "Bot: Failed to extract the table.",
        "de": "Bot: Tabelle konnte nicht extrahiert werden.",
        "fr": "Bot: Impossible d'extraire le tableau.",
    },
    "exit": {
        "ru": "\nВыход.",
        "en": "\nExit.",
        "de": "\nBeenden.",
        "fr": "\nSortie.",
    },
    "error": {
        "ru": "Ошибка:",
        "en": "Error:",
        "de": "Fehler:",
        "fr": "Erreur:",
    },
    "tg_greeting": {
        "ru": "Привет! Выберите режим работы бота:",
        "en": "Hi! Choose the bot mode:",
        "de": "Hallo! Wählen Sie den Bot-Modus:",
        "fr": "Bonjour! Choisissez le mode du bot :",
    },
    "tg_moral_block": {
        "ru": "К сожалению, ваш запрос нарушает мои моральные принципы и не будет обработан.",
        "en": "Sorry, your request violates my moral principles and cannot be processed.",
        "de": "Leider verstößt Ihre Anfrage gegen meine moralischen Prinzipien.",
        "fr": "Désolé, votre demande enfreint mes principes moraux.",
    },
    "tg_press_button": {
        "ru": "Пожалуйста, нажмите кнопку для выбора режима.",
        "en": "Please press a button to choose a mode.",
        "de": "Bitte drücken Sie eine Taste, um einen Modus zu wählen.",
        "fr": "Veuillez appuyer sur un bouton pour choisir un mode.",
    },
    "tg_enter_query": {
        "ru": "Введите ваш запрос:",
        "en": "Enter your query:",
        "de": "Geben Sie Ihre Anfrage ein:",
        "fr": "Entrez votre requête :",
    },
    "tg_next": {
        "ru": "Выберите следующий режим или повторите запрос:",
        "en": "Choose the next mode or repeat the query:",
        "de": "Wählen Sie den nächsten Modus oder wiederholen Sie die Anfrage:",
        "fr": "Choisissez le mode suivant ou répétez la requête :",
    },
    "tg_started": {
        "ru": "Telegram-бот запущен.",
        "en": "Telegram bot started.",
        "de": "Telegram-Bot gestartet.",
        "fr": "Bot Telegram démarré.",
    },
    "query_type_undetected": {
        "ru": "Не удалось определить тип запроса. Выберите режим вручную.",
        "en": "Unable to determine the type of request. Please select the mode manually.",
        "de": "Der Anfragetyp konnte nicht bestimmt werden. Bitte wählen Sie den Modus manuell aus.",
        "fr": "Impossible de déterminer le type de requête. Veuillez sélectionner le mode manuellement.",
    },
    "lang_clf_not_trained": {
        "ru": "Классификатор языка ещё не обучен",
        "en": "Language classifier has not been trained yet",
        "de": "Der Sprachklassifikator wurde noch nicht trainiert",
        "fr": "Le classificateur de langue n’a pas encore été entraîné",
    },
    "lang_clf_training": {
        "ru": "Обучение классификатора языка...",
        "en": "Training language classifier...",
        "de": "Training des Sprachklassifikators...",
        "fr": "Entraînement du classificateur de langue...",
    },
    "lang_clf_trained": {
        "ru": "Классификатор языка обучен\n",
        "en": "Language classifier has been trained\n",
        "de": "Der Sprachklassifikator wurde trainiert\n",
        "fr": "Le classificateur de langue a été entraîné\n",
    },
    "moral_not_trained": {
        "ru": "Моральный фильтр ещё не обучен",
        "en": "Moral filter has not been trained yet",
        "de": "Der Moralfilter wurde noch nicht trainiert",
        "fr": "Le filtre moral n’a pas encore été entraîné",
    },
    "moral_training": {
        "ru": "Обучение морального фильтра...",
        "en": "Training moral filter...",
        "de": "Training des Moralfilters...",
        "fr": "Entraînement du filtre moral...",
    },
    "moral_trained": {
        "ru": "Моральный фильтр обучен\n",
        "en": "Moral filter has been trained\n",
        "de": "Der Moralfilter wurde trainiert\n",
        "fr": "Le filtre moral a été entraîné\n",
    },
    "domain_not_trained": {
        "ru": "Классификатор домена ещё не обучен",
        "en": "Domain classifier has not been trained yet",
        "de": "Domain-Klassifikator wurde noch nicht trainiert",
        "fr": "Le classificateur de domaine n'a pas encore été entraîné",
    },
    "domain_training": {
        "ru": "Обучение классификатора домена...",
        "en": "Training domain classifier...",
        "de": "Training des Domain-Klassifikators...",
        "fr": "Entraînement du classificateur de domaine...",
    },
    "domain_trained": {
        "ru": "Классификатор домена обучен\n",
        "en": "Domain classifier has been trained\n",
        "de": "Domain-Klassifikator wurde trainiert\n",
        "fr": "Le classificateur de domaine a été entraîné\n",
    },
    "choose_collection": {
        "ru": "Выберите коллекцию для добавления:",
        "en": "Choose a collection to add text to:",
        "de": "Wählen Sie eine Sammlung zum Hinzufügen von Text:",
        "fr": "Choisissez une collection pour ajouter du texte :",
    },
    "all_collections": {
        "ru": "4 – Во все коллекции сразу",
        "en": "4 – To all collections",
        "de": "4 – In alle Sammlungen",
        "fr": "4 – Dans toutes les collections",
    },
    "your_choice": {
        "ru": "Ваш выбор: ",
        "en": "Your choice: ",
        "de": "Ihre Wahl: ",
        "fr": "Votre choix : ",
    },
    "text_added_all": {
        "ru": "Текст добавлен во все коллекции.",
        "en": "Text added to all collections.",
        "de": "Text wurde zu allen Sammlungen hinzugefügt.",
        "fr": "Le texte a été ajouté à toutes les collections.",
    },
    "text_added": {
        "ru": "Текст добавлен в выбранную коллекцию.",
        "en": "Text added to the selected collection.",
        "de": "Text wurde zur ausgewählten Sammlung hinzugefügt.",
        "fr": "Le texte a été ajouté à la collection sélectionnée.",
    },
    "table_choice_prompt": {
        "ru": "Выберите источник таблицы:\n1 – Таблица из базы\n2 – Таблица из документа\n",
        "en": "Choose table source:\n1 – Table from the database\n2 – Table from the document\n",
        "de": "Wählen Sie die Tabellenquelle:\n1 – Tabelle aus der Datenbank\n2 – Tabelle aus dem Dokument\n",
        "fr": "Choisissez la source du tableau:\n1 – Tableau depuis la base\n2 – Tableau depuis le document\n",
    },
    "summary_prompt": {
        "ru": "Сделай краткий пересказ текста на русском языке. Отвечай только текстом, без лишних пояснений.",
        "en": "Summarize the text in English. Reply only with text, no extra explanations.",
        "de": "Fasse den Text auf Deutsch zusammen. Antworte nur mit Text, ohne zusätzliche Erklärungen.",
        "fr": "Fais un résumé du texte en français. Réponds uniquement par le texte, sans explications supplémentaires.",
    },
    "summary_fail": {
        "ru": "Не удалось создать пересказ или тэги.",
        "en": "Failed to generate summary or tags.",
        "de": "Fehler beim Erstellen von Zusammenfassung oder Tags.",
        "fr": "Échec de la génération du résumé ou des tags.",
    },
    "summary_result": {
        "ru": "Пересказ:\n{summary}\n\nТэги: {tags}\n",
        "en": "Summary:\n{summary}\n\nTags: {tags}\n",
        "de": "Zusammenfassung:\n{summary}\n\nTags: {tags}\n",
        "fr": "Résumé:\n{summary}\n\nTags: {tags}\n",
    },
}
