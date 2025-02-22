import sqlite3
import json


def lookup_products_by_ids(product_ids):
    conn = sqlite3.connect('Data/database.db')
    cursor = conn.cursor()
    # Формуємо рядок з плейсхолдерами, рівний кількості ідентифікаторів
    placeholders = ','.join('?' for _ in product_ids)
    query = f"""
        SELECT ProductID, ProductTitle, ProductURL, ProductImage
        FROM StockTable
        WHERE ProductID IN ({placeholders})
    """
    cursor.execute(query, product_ids)
    rows = cursor.fetchall()
    conn.close()

    results = {}
    # Записуємо знайдені товари у словник
    for row in rows:
        pid, title, url, image = row
        results[pid] = {
            "id": pid,
            "row_index": title,
            "website_link": url,
            "image_link": image
        }
    # # Для кожного ID, який не знайдено, додаємо повідомлення про помилку
    # for pid in product_ids:
    #     if pid not in results:
    #         results[pid] = {"error": "Товар не знайдено"}

    return json.dumps(results, ensure_ascii=False, indent=2)


# Приклад використання:
if __name__ == "__main__":
    input_ids = input("Введіть ID товарів через кому: ")
    # Розділяємо вхідний рядок на список ID та обрізаємо зайві пробіли
    id_list = [i.strip() for i in input_ids.split(",") if i.strip()]
    print(lookup_products_by_ids(id_list))
