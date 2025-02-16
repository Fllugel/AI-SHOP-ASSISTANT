import sqlite3
import json

import sqlite3
import json


def lookup_product_by_id(product_id):
    """
    Tool Name: lookup_product_by_id

    Description:
    This tool searches the "products" table in the 'database.db' SQLite database for a product with the specified product_id.
    If the product is found, it returns a JSON-formatted string containing:
      - "id": The product's ID.
      - "row_index": The index (rowid) of the product in the database.
      - "website_link": The URL link to the product's page on the website.
      - "image_link": The URL link to the product's image.

    If the product is not found, it returns a JSON-formatted string with an "error" key indicating that the product was not found.

    Parameters:
    - product_id: The unique identifier of the product to search for in the database.

    Returns:
    - A JSON string with product details if found, or an error message if not.
    """
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
