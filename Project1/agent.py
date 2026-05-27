from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Debug check
print("API KEY:", os.getenv("GROQ_API_KEY"))

import base64
import json
import sqlite3
from typing import Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating


# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

vision_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# ---------------------------------------------------------------------------
# Database Path
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None
) -> str:
    """
    Search products from the database.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = """
        SELECT id, name, category, price, description, is_organic
        FROM products
        WHERE 1=1
    """

    params = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)

    rows = cursor.fetchall()

    conn.close()

    products = [
        {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "description": row[4],
            "is_organic": bool(row[5]),
        }
        for row in rows
    ]

    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get product rating.
    """

    result = get_product_rating(product_id)

    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place order for product.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, price FROM products WHERE id = ?",
        (product_id,)
    )

    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: Product with ID {product_id} not found."

    name, price = row

    cursor.execute(
        """
        INSERT INTO orders (product_id, product_name, price)
        VALUES (?, ?, ?)
        """,
        (product_id, name, price)
    )

    order_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed!\n"
        f"Product: {name}\n"
        f"Price: ${price:.2f}\n"
        f"Delivery: 3-5 business days."
    )


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze product image.
    """

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")

    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{image_data}"
                },
            },
            {
                "type": "text",
                "text": (
                    "Look at this product image and extract key attributes.\n"
                    "Return ONLY JSON with:\n"
                    "- product_type\n"
                    "- search_query\n"
                    "- is_organic\n"
                    "- description"
                ),
            },
        ]
    )

    response = vision_llm.invoke([message])

    return response.content


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    tools=[
        search_products,
        get_rating,
        checkout,
        describe_product_image
    ],
    model=llm,
    system_prompt=(
        "You are a helpful shopping assistant.\n\n"

        "IMAGE SEARCH FLOW:\n"
        "1. Use describe_product_image for uploaded images.\n"
        "2. Use extracted search_query with search_products.\n"
        "3. Continue browsing flow.\n\n"

        "BROWSING FLOW:\n"
        "1. Use search_products.\n"
        "2. Use get_rating for each product.\n"
        "3. Show products in numbered format.\n"
        "4. Include product ID.\n"
        "5. Never order automatically.\n\n"

        "ORDER FLOW:\n"
        "1. Only order after explicit confirmation.\n"
        "2. Use checkout(product_id).\n"
        "3. Confirm order."
    )
)


# ---------------------------------------------------------------------------
# Test Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want organic honey under $20 with rating above 4.5"
                    ),
                }
            ]
        }
    )

    print(result["messages"][-1].content)