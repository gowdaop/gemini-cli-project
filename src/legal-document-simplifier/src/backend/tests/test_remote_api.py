import os, pytest, requests, time, json

BASE = os.getenv("REMOTE_BASE_URL", "http://localhost:8000").rstrip("/")
KEY  = os.getenv("API_KEY", "my-strong-secret")
s = requests.Session(); s.headers["x-api-key"]=KEY

def url(p): return f"{BASE}{p}"

def wait_ready():
    for _ in range(10):
        try:
            if s.get(url("/healthz"), timeout=5).status_code==200:
                return
        except Exception: pass
        time.sleep(2)
    pytest.skip("API not up")

@pytest.fixture(scope="session", autouse=True)
def _ready(): wait_ready()

def test_upload_analyze_chat_smoke(tmp_path):
    # 1 upload a tiny PNG
    f = tmp_path/"tiny.png"; f.write_bytes(b"\x89PNG\r\n\x1a\n")
    r = s.post(url("/upload"), files={"file":("tiny.png", f.open("rb"), "image/png")})
    assert r.status_code==200
    ocr = r.json()["ocr"]

    # 2 analyze with top-k 2
    r = s.post(url("/analyze"), json={"ocr": ocr, "top_k": 2})
    assert r.status_code==200
    body = r.json()
    assert "clauses" in body and "risks" in body

    # 3 chat about the doc
    r = s.post(url("/chat"), json={"question":"Summarize liabilities"})
    assert r.status_code==200
    assert r.json()["answer"]

