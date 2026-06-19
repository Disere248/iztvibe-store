import math
from groq import Groq

def tokenize(text):
    """Очистка и разбиение текста на токены."""
    return [word.lower().strip(".,!?\"()—") for word in text.split() if len(word) > 2]

def calculate_cosine_similarity(query, doc_text):
    """Расчет косинусного сходства между запросом пользователя и описанием товара."""
    query_words = tokenize(query)
    doc_words = tokenize(doc_text)
    
    if not query_words or not doc_words:
        return 0.0
        
    all_words = set(query_words + doc_words)
    
    query_vector = [query_words.count(word) for word in all_words]
    doc_vector = [doc_words.count(word) for word in all_words]
    
    dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
    
    query_len = math.sqrt(sum(q**2 for q in query_vector))
    doc_len = math.sqrt(sum(d**2 for d in doc_vector))
    
    if query_len == 0 or doc_len == 0:
        return 0.0
        
    return dot_product / (query_len * doc_len)

def get_relevant_products(user_query, items, top_n=3):
    """Поиск наиболее подходящих товаров на основе локального векторного сходства."""
    scored_products = []
    for item in items:
        full_text = f"{item['name']} {item['category']} {item['material']} {item['description']}"
        score = calculate_cosine_similarity(user_query, full_text)
        if score > 0:
            scored_products.append((score, item))
            
    scored_products.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored_products[:top_n]]

def ask_groq_stylist(api_key, user_query, relevant_items, all_items, chat_history):
    """Генерация строго структурированных рекомендаций без галлюцинаций валют."""
    client = Groq(api_key=api_key)
    
    # Формирую базу данных с жесткой фиксацией цен в тенге
    catalog_dump = ""
    for item in all_items:
        catalog_dump += f"Товар: {item['name']} | Точная Цена: {item['price']} ₸ | Материал: {item['material']} | Описание: {item['description']}\n"
    
    messages = [
        {
            "role": "system",
            "content": (
                f"Ты — высококлассный, уверенный в себе ИИ-стилист бутика IZTVibe Store.\n"
                f"Ты общаешься с клиентом профессионально, лаконично и авторитетно, без лишней воды.\n\n"
                f"ЭТАЛОННЫЙ КАТАЛОГ С ЦЕНАМИ:\n{catalog_dump}\n"
                f"ИНСТРУКЦИЯ ПО ФОРМИРОВАНИЮ ОТВЕТА:\n"
                f"1. Цены пиши СТРОГО в тенге (₸) в полном соответствии с каталогом. Забудь про рубли и доллары.\n"
                f"2. Если клиент спрашивает, хорошая ли вещь, не отвечай шаблонно 'нет информации'. Используй данные из каталога, чтобы подтвердить качество: подчеркни, что это плотная ткань, премиальный хлопок, плотный деним или влагозащитный нейлон.\n"
                f"3. Твоя цель - помочь клиенту собрать стильный образ, аргументируя выбор деталями кроя и материала из каталога.\n"
                f"4. Не придумывай несуществующие цифры или комплектации."
            )
        }
    ]
    
    # Подгружаю историю (последние 4 реплики)
    max_history_depth = 4
    for role, text in chat_history[-max_history_depth:]:
        messages.append({"role": role, "content": text})
        
    messages.append({
        "role": "user",
        "content": f"Запрос пользователя: '{user_query}'"
    })
    
    # Ставлю температуру в 0.0 чтобы не писал лишнее
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant",
        temperature=0.1
    )
    return chat_completion.choices[0].message.content
