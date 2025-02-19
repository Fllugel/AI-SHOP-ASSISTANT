import sqlite3
import json


def lookup_product_by_id(product_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT ProductTitle, ProductURL, ProductImage FROM StockTable WHERE PRODUCT_ID = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        row_index, website_link, image_link = row
        response = {
            "id": product_id,
            "row_index": row_index,
            "website_link": website_link,
            "image_link": image_link
        }
    else:
        response = {"error": "Товар не знайдено"}

    return json.dumps(response, ensure_ascii=False, indent=2)


# Приклад використання:
if __name__ == "__main__":
    test_id = input("Введіть ID товару: ")
    print(lookup_product_by_id(test_id))
