from langchain_core.tools import tool
from Tools.product_lookup_tool import lookup_product_by_id
from Tools.shop_info_tool import shop_info
from Tools.holiday_info_tool import holiday_info
from Tools.sql_db_tool import find_data_in_db


@tool("final_answer")
def final_answer(answer: str):
    """
    Tool to returns the final answer to the user.
    Use this tool only after all necessary processing is complete.
    It returns the final response to the user.
    """
    return answer.strip()


@tool("holiday_info_tool")
def holiday_info_tool(key: str) -> str:
    """
    This tool returns a list of gift categories suitable for the holiday
    that best matches the input key.
    """
    # Query the FAISS index for the most similar document.
    return holiday_info(key)


@tool("product_lookup_tool")
def product_lookup_tool(key: str):
    """
    This tool searches the "products" table in the 'database.db' SQLite database for a product with the specified product_id.
    """
    return lookup_product_by_id(key)


@tool("shop_info_tool")
def shop_info_tool() -> str:
    """
    Tool returns all information about the company.
    """
    return shop_info()


@tool("sql_db_tool")
def sql_db_tool(question: str, history: list) -> str:
    """Tool for searching the store's database. The database contains the following data about products: product article, product category, product sub-category, product name, available quantity in the store, and price per unit."""
    return find_data_in_db(question, history)


tools = [sql_db_tool, shop_info_tool, holiday_info_tool, product_lookup_tool, final_answer]

tool_str_to_func = {
    "final_answer": final_answer,
    "sql_db_tool": sql_db_tool,
    "shop_info_tool": shop_info_tool,
    "holiday_info_tool": holiday_info_tool,
    "product_lookup_tool": product_lookup_tool,
}
