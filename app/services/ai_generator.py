import json
import logging
import re
from app.core.region_loader import load_region
from app.core.config import AI_PROVIDER, AI_MODEL, AI_API_KEY
import openai
import requests


# -------------------------------
# Low-level AI call handler
# -------------------------------
def call_ai_model(prompt: str) -> str:
    """
    Calls the configured AI provider and returns the text output.
    Supports OpenAI, Google Gemini, and DeepSeek.
    """
    provider = str(AI_PROVIDER).strip().lower() if AI_PROVIDER else None
    model = str(AI_MODEL).strip() if AI_MODEL else None

    if not provider:
        raise ValueError("AI_PROVIDER not set in config")

    if provider == "openai":
        if not AI_API_KEY:
            raise ValueError("AI_API_KEY not set for OpenAI")
        openai.api_key = AI_API_KEY
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    elif provider == "google-gemini":
        if not AI_API_KEY:
            raise ValueError("AI_API_KEY not set for Google Gemini")
        try:
            import google.generativeai as genai
            genai.configure(api_key=AI_API_KEY)
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            raise Exception(f"Failed to call Google Gemini API: {e}")

    elif provider == "deepseek":
        if not AI_API_KEY:
            raise ValueError("AI_API_KEY not set for DeepSeek")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {AI_API_KEY}"}
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error(f"DeepSeek API error: {e}")
            raise Exception(f"Failed to call DeepSeek API: {e}")

    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {AI_PROVIDER}")


def extract_json_from_text(text: str) -> dict:
    """
    Extracts and parses JSON from text that might contain extra content.
    Handles various formatting issues that AI models might produce.
    """
    # First, try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object or array in the text
    json_patterns = [
        r'\{.*\}',  # JSON object
        r'\[.*\]',  # JSON array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                # Try with common fixes
                fixes = [
                    # Fix 1: Replace single quotes with double quotes
                    lambda s: s.replace("'", '"'),
                    # Fix 2: Remove trailing commas
                    lambda s: re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', s)),
                    # Fix 3: Escape unescaped quotes within strings
                    lambda s: re.sub(r'(?<!\\)"', '\\"', s),
                    # Fix 4: Handle multiline strings by escaping newlines
                    lambda s: s.replace('\n', '\\n').replace('\r', '\\r'),
                ]

                for fix in fixes:
                    try:
                        fixed_json = fix(match)
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        continue

    # If all else fails, try a more aggressive cleanup
    try:
        # Remove any text before first { or [
        start_chars = ['{', '[']
        start_pos = min([text.find(char) for char in start_chars if text.find(char) != -1] or [0])
        cleaned_text = text[start_pos:]

        # Remove any text after balanced brackets
        bracket_stack = []
        end_pos = 0
        for i, char in enumerate(cleaned_text):
            if char in ['{', '[']:
                bracket_stack.append(char)
            elif char in ['}', ']']:
                if bracket_stack:
                    bracket_stack.pop()
                    if not bracket_stack:
                        end_pos = i + 1
                        break

        if end_pos > 0:
            final_json = cleaned_text[:end_pos]
            return json.loads(final_json)
    except Exception as e:
        logging.error(f"Advanced JSON extraction failed: {e}")

    raise ValueError("Could not extract valid JSON from AI response")


# -------------------------------
# Main Exam Generator
# -------------------------------
def generate_exam_questions(
        region: str,
        subject: str,
        class_level: str,
        topics: list,
        difficulty: str,
        num_objectives: int,
        num_essays: int,
        essay_style: str = "single"
):
    """
    Generates exam questions (objective + essay) using AI.
    Returns structured data with questions and answers.
    Ensures JSON output from AI is strict and parseable.
    """
    # Load region data
    region_data = load_region(region)

    # Validate class level
    level_data = None
    class_level_lower = class_level.lower()
    for lvl_group in region_data["levels"].values():
        if any(cl.lower() == class_level_lower for cl in lvl_group.get("grades", [])):
            level_data = lvl_group
            break
    if not level_data:
        raise ValueError(f"Class level '{class_level}' not found in region '{region}'")

    # Validate topics
    for t in topics:
        if not t.strip():
            raise ValueError("Empty topic submitted")

    # -------------------------------
    # Build AI prompt with strict JSON instructions
    # -------------------------------
    prompt = f"""
You are an expert exam setter.
Generate exam questions strictly in JSON format.
Do NOT include any explanatory text outside JSON.
Use the following structure:

{{
  "objectives": [
    {{
      "question": "Question text",
      "options": {{"a": "Option A", "b": "Option B", "c": "Option C", "d": "Option D"}},
      "answer": "a|b|c|d"
    }}
  ],
  "essays": [
    {{
      "question": "Main question text",
      "sub_questions": ["Optional sub-question texts"],
      "answer": "Full model answer"
    }}
  ]
}}

IMPORTANT:
- Escape all newlines inside strings as \\\\n.
- Escape all quotes inside strings as \\\\".
- Output ONLY JSON, no extra formatting or text.
- FOR ESSAYS: The 'answer' field MUST contain the full model answer divided and sequentially lettered to match the question structure. If the essay is question '1', the answer for the main part must be prefixed with '1a.', the answer for the first sub-question must be prefixed with '1b.', the second sub-question with '1c.', and so on.

Subject: {subject}
Class Level: {class_level}
Difficulty: {difficulty}
Topics: {', '.join(topics)}
Num Objectives: {num_objectives}
Num Essays: {num_essays}
Essay Style: {essay_style}
"""

    # -------------------------------
    # Call AI
    # -------------------------------
    output_text = call_ai_model(prompt)

    # -------------------------------
    # Parse JSON with robust extraction
    # -------------------------------
    try:
        data = extract_json_from_text(output_text)
    except Exception as e:
        logging.error(f"Failed to parse AI response as JSON: {e}")
        logging.error(f"Raw AI output: {output_text}")
        raise ValueError(f"AI returned invalid JSON format: {str(e)}")

    # -------------------------------
    # Validate and build structured output
    # -------------------------------
    objectives = []
    obj_answers = []

    for i, obj in enumerate(data.get("objectives", [])):
        if not isinstance(obj, dict):
            logging.warning(f"Skipping invalid objective at index {i}")
            continue

        try:
            objectives.append({
                "question": obj.get("question", "").strip(),
                "options": {k.lower(): str(v).strip() for k, v in obj.get("options", {}).items()}
            })
            # Format objective answers as "1. a", "2. b", etc.
            answer_text = f"{len(obj_answers) + 1}. {obj.get('answer', '').strip().upper()}"
            obj_answers.append({"type": "objective", "answer": answer_text})
        except Exception as e:
            logging.warning(f"Failed to process objective at index {i}: {e}")
            continue

    essays = []
    essay_answers = []

    for i, essay in enumerate(data.get("essays", [])):
        if not isinstance(essay, dict):
            logging.warning(f"Skipping invalid essay at index {i}")
            continue

        try:
            essays.append({
                "question": essay.get("question", "").strip(),
                "sub_questions": [str(sq).strip() for sq in essay.get("sub_questions", [])]
            })

            # Format essay answers with proper numbering
            essay_number = len(essay_answers) + 1
            answer_content = essay.get("answer", "").strip()

            if essay_style.lower() == "nested" and essay.get("sub_questions"):
                # For nested essays, format with sub-question numbering (1a, 1b, etc.)
                formatted_answer = []
                sub_questions = essay.get("sub_questions", [])
                answer_lines = answer_content.split('\n')

                # Try to match answer lines with sub-questions
                for j, sub_q in enumerate(sub_questions):
                    prefix = f"{essay_number}{chr(97 + j)}."  # 1a, 1b, 1c, etc.
                    # Find the corresponding answer part (this is heuristic)
                    if j < len(answer_lines) and answer_lines[j].strip():
                        formatted_answer.append(f"{prefix} {answer_lines[j].strip()}")
                    else:
                        # If we can't match line by line, just include the full answer with sub-question numbers
                        formatted_answer.append(f"{prefix} [Answer for sub-question {j + 1}]")

                # If we have remaining answer content, append it
                if len(answer_lines) > len(sub_questions):
                    for k in range(len(sub_questions), len(answer_lines)):
                        if answer_lines[k].strip():
                            sub_prefix = f"{essay_number}{chr(97 + k)}." if k < 26 else f"{essay_number}.{k + 1}"
                            formatted_answer.append(f"{sub_prefix} {answer_lines[k].strip()}")

                essay_answer_text = "\n\n".join(formatted_answer)
            else:
                # For single essays, just use the number
                essay_answer_text = f"{essay_number}. {answer_content}"

            essay_answers.append({"type": "essay", "answer": essay_answer_text})

        except Exception as e:
            logging.warning(f"Failed to process essay at index {i}: {e}")
            continue

    # Combine all answers
    all_answers = obj_answers + essay_answers

    return {
        "objectives": objectives,
        "essays": essays,
        "answers": all_answers,
        "raw_output": output_text
    }