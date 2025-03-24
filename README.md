# AI-SHOP-ASSISTANT

## Overview
AI-SHOP-ASSISTANT is a Python-based project that provides an AI-powered assistant for shop-related queries. It uses FastAPI for the backend and integrates various tools for handling different types of information.

## Requirements
- Python 3.8+
- pip

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ai-shop-assistant.git
    cd ai-shop-assistant
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application
1. Start the FastAPI server:
    ```sh
    uvicorn api:app --reload
    ```

2. The server will be running at `http://127.0.0.1:8000`.

## Endpoints
### Chat Endpoint
- **URL:** `/chat`
- **Method:** `POST`
- **Description:** Handles user queries and returns responses from the AI assistant.
- **Request Body:**
    ```json
    {
        "user_id": "string",
        "input": "string"
    }
    ```
- **Response:**
    ```json
    {
        "response_message": "string"
    }
    ```

### Clear History Endpoint
- **URL:** `/clear_history`
- **Method:** `POST`
- **Description:** Clears the chat history for a given user.
- **Request Body:**
    ```json
    {
        "user_id": "string"
    }
    ```
- **Response:**
    ```json
    {
        "message": "Chat history cleared."
    }
    ```

## Additional Information
- Ensure that the `origins` list in `api.py` is updated with your front-end domain(s) to allow CORS.
- The project includes tools for handling shop information, product lookup, holiday information, and SQL database queries.