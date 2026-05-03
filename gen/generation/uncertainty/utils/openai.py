import os
import hashlib
from pathlib import Path
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type
from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError

project_root = Path(__file__).resolve().parents[4]
load_dotenv()
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)

subscription_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

client = None
if subscription_key and azure_endpoint:
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        api_key=subscription_key,
    )

class MissingAPIKeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass

@retry(
    retry=retry_if_not_exception_type(MissingAPIKeyError),
    wait=wait_random_exponential(min=1, max=10)
)
def predict(prompt, model='gpt-5-mini'): 
    """
    Predict with Azure OpenAI models. 
    Modified for o1-mini/gpt-5-mini compatibility (no temperature arg).
    """

    if client is None or not client.api_key:
        raise MissingAPIKeyError(
            'Need to provide Azure OpenAI credentials via AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.'
        )

    if isinstance(prompt, str):
        messages = [{'role': 'user', 'content': prompt}]
    else:
        messages = prompt

    # Debug print
    # print(f"Messages: {messages}")

    try:
        # -------------------------------------------------------
        # Key compatibility changes:
        # 1) Remove temperature (unsupported on some reasoning models)
        # 2) Use max_completion_tokens instead of max_tokens
        # -------------------------------------------------------
        output = client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort="low",
            max_completion_tokens=512,  # Increase if needed (e.g., 16384)
        )
        
        # print(f"Output: {output}")
        return output.choices[0].message.content

    except BadRequestError as e:
        print("⚠️ BadRequestError caught:", e)
        # Print detailed error info for easier debugging
        print(e.response.json() if hasattr(e, 'response') else "No response body")
        return None

    except Exception as e:
        print("⚠️ Unexpected exception in predict():", type(e).__name__, e)
        raise

def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
