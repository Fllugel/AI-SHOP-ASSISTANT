import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from typing_extensions import Annotated, TypedDict
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from config import SQL_DB_TOOL_PRODUCT_DB_URI, SQL_DB_TOOL_LLM_MODEL_NAME, SQL_DB_TOOL_LLM_MODEL_TEMPERATURE, \
    SQL_DB_TOOL_TOP_K, \
    SQL_DB_TOOL_MAX_ATTEMPTS

load_dotenv()

db = SQLDatabase.from_uri(SQL_DB_TOOL_PRODUCT_DB_URI)
db.run("PRAGMA case_sensitive_like=OFF;")

tool_prompt = PromptTemplate.from_template("""
Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 
Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. 
Never query for all the columns from a specific table, only ask for the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. 

If a client asks for a product recommendation without a detailed description, choose a random product from the database using RANDOM. 
When searching for a specific product, and you do not find the product in the database, use multiple synonyms or rephrased versions of the product name in query. 
ALWAYS make queries in noun infinitive form when searching for a specific product. 

DISTINCT must come directly after SELECT as in example: "SELECT DISTINCT column1, column2, ... ";

Only use the following tables:
{table_info}

Question: {input}
""")


# Define the State object structure
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    history: list
    empty_queries: list


# Define the QueryOutput structure
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


llm = ChatOpenAI(model=SQL_DB_TOOL_LLM_MODEL_NAME, temperature=SQL_DB_TOOL_LLM_MODEL_TEMPERATURE,
                 openai_api_key=os.getenv("GPT_API_KEY"))


# Function to write the query
def write_query(state: State):
    prompt = tool_prompt.invoke(
        {
            "dialect": db.dialect,
            "top_k": SQL_DB_TOOL_TOP_K,
            "table_info": db.get_table_info(),
            "input": state["question"],
            "history": state["history"]
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)

    query = result["query"]
    if "SELECT ProductID" not in query.upper():
        query = query.replace("SELECT", "SELECT ProductID,", 1)

    return {"query": query}


# Function to execute the query
def execute_query(state: State):
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


# Function to generate the answer
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}\n'
        f'History: {state["history"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


def rephrase_query(state: State) -> str:
    synonyms_prompt = (
        f"""The SQL query '{state['query']}' returned no results for the product search based on the question: '{state['question']}'. 
Please provide an alternative syntactically correct SQL query using synonyms, 
different phrasing of the product name, or try searching in english and in ukrainian.
Ensure that the query uses the correct column names and limits the results to at most {SQL_DB_TOOL_TOP_K}. 
DISTINCT must come directly after SELECT.
""")
    response = llm.invoke(synonyms_prompt)
    new_query = response.content.strip()
    if "SELECT PRODUCT_ID" not in new_query.upper():
        new_query = new_query.replace("SELECT", "SELECT ProductID,", 1)
    return new_query


def find_data_in_db(question: str, history: list) -> str:
    try:
        state = {"question": question, "history": history, "empty_queries": []}
        state.update(write_query(state))
        state.update(execute_query(state))

        # Retry generating and executing alternative SQL queries until a non-empty result is obtained or the maximum attempts are reached.
        attempt = 0
        while (not state["result"] or state["result"] == "") and attempt < SQL_DB_TOOL_MAX_ATTEMPTS:
            new_query = rephrase_query(state)
            if new_query in state["empty_queries"]:
                attempt += 1
                continue
            state["empty_queries"].append(new_query)
            state["query"] = new_query
            state.update(execute_query(state))
            attempt += 1

        # Optionally, generate an answer:
        # state.update(generate_answer(state))
        return state["result"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

