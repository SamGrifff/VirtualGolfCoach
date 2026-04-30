import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def generate_fallback_feedback(result: dict) -> str:
    strongest = max(result["phases"], key=lambda p: p["score"])
    weakest = min(result["phases"], key=lambda p: p["score"])

    return (
        f"Your strongest phase was {strongest['name']} ({strongest['score']}/100). "
        f"The biggest improvement area was {weakest['name']} ({weakest['score']}/100). "
        f"Overall swing score: {result['overall_score']}/100 ({result['grade']}). "
        f"Priority focus: {weakest['feedback']}"
    )


def generate_ai_feedback(result: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return generate_fallback_feedback(result)

    strongest = max(result["phases"], key=lambda p: p["score"])
    weakest = min(result["phases"], key=lambda p: p["score"])

    compact_result = {
        "overall_score": result["overall_score"],
        "grade": result["grade"],
        "worst_frame": result["worst_frame"],
        "strongest_phase": strongest["name"],
        "weakest_phase": weakest["name"],
        "metrics": result["metrics"],
        "phases": [
            {
                "name": p["name"],
                "score": p["score"],
                "feedback": p["feedback"],
            }
            for p in result["phases"]
        ],
    }

    prompt = f"""
You are an expert golf coach.

Using the swing analysis data below, write:
1. A short coaching summary in 2-3 sentences
2. Three actionable coaching bullet points
3. One priority improvement area with tips on how to improve it

Rules:
- Be concise and practical
- Sound like a golf coach
- Do not invent measurements or claims not present in the data
- Base your response only on the analysis data provided

Analysis data:
{json.dumps(compact_result, indent=2)}
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

        return generate_fallback_feedback(result)

    except Exception:
        return generate_fallback_feedback(result)