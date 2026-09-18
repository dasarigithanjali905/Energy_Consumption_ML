import os
from typing import Any, Dict

import requests


API_URL = "https://router.huggingface.co/v1/chat/completions"


def _get_headers() -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def query(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = _get_headers()
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def generate_ai_response(ticket_text: str) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional customer support assistant.",
            },
            {"role": "user", "content": ticket_text},
        ],
        "model": "deepseek-ai/DeepSeek-R1:novita",
    }
    try:
        data = query(payload)
        return data["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return f"LLM Error: {str(e)}"

