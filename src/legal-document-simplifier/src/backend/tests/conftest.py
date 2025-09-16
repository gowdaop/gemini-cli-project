import pytest
from fastapi.testclient import TestClient
from src.backend.main import app

@pytest.fixture(scope="module")
def mock_api_key():
    return "test-api-key"

@pytest.fixture(scope="module")
def sample_ocr_text():
    return {
        "full_text": "This is a sample document.",
        "blocks": [
            {
                "text": "This is a sample document.",
                "span": {
                    "page": 1,
                    "start_line": 1,
                    "end_line": 1
                }
            }
        ]
    }

@pytest.fixture(scope="module")
async def async_client():
    async with TestClient(app, base_url="http://test") as client:
        yield client

@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://test") as c:
        yield c
