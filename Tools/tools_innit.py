from typing import List

from langchain_core.tools import tool
from Tools.product_lookup_tool import lookup_products_by_ids
from Tools.shop_info_tool import shop_info
from Tools.holiday_info_tool import holiday_info
from Tools.sql_db_tool import find_data_in_db


@tool("holiday_info_tool")
def holiday_info_tool(key: str) -> str:
    """
    This tool returns a list of gift categories suitable for the holiday
    that best matches the input key.
    """
    # Query the FAISS index for the most similar document.
    return holiday_info(key)


@tool("product_lookup_tool")
def product_lookup_tool(product_ids: List[str]):
    """
    This tool accepts product IDs and generates a JSON for the frontend with images and product names. The output from this tool is sent directly to the frontend, so it does not return anything.
    """
    return lookup_products_by_ids(product_ids)


@tool("shop_info_tool")
def shop_info_tool() -> str:
    """
    Tool returns all information about the company.
    """
    return shop_info()


@tool("sql_db_tool")
def sql_db_tool(question: str, history: list) -> str:
    """Tool for searching the store's database. The database contains the following data about products"""
    return find_data_in_db(question, history)


tools = [sql_db_tool, shop_info_tool, holiday_info_tool, product_lookup_tool]

tool_str_to_func = {
    "sql_db_tool": sql_db_tool,
    "shop_info_tool": shop_info_tool,
    "holiday_info_tool": holiday_info_tool,
    "product_lookup_tool": product_lookup_tool,
}
