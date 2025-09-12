import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from typing import List

_GEMINI = GenerativeModel("gemini-2.0-flash")
ASYNC_TIMEOUT = 20  # s

async def generate_clause_recommendations(
    clause_text: str, risk_level: str, risk_score: float
) -> List[str]:
    """Return 2-4 bullet-style remediation tips for one clause."""
    prompt = (
        "You are a senior commercial lawyer.\n"
        f'Clause:\n"""\n{clause_text[:1_500]}\n"""\n'
        f"Risk level: {risk_level.upper()} (score {risk_score:0.2f}).\n"
        "Give 2-4 short, practical recommendations to mitigate this risk. "
        "Start each line with a hyphen."
    )
    try:
        resp = await _GEMINI.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.3,
                max_output_tokens=256,
            ),
            timeout=ASYNC_TIMEOUT,
        )
        return [
            ln.lstrip("- ").strip()
            for ln in (resp.text or "").splitlines()
            if ln.strip()
        ]
    except Exception as err:
        return [f"⚠️ Vertex AI recommendations unavailable ({err})"]
