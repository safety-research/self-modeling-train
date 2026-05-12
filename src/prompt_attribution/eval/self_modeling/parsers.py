"""
Module: prompt_attribution/eval/self_modeling/parsers.py

Response parsers for introspection evals. All evals use JSON format:
  {"reasoning": "<step-by-step reasoning>", "answer": <typed answer>}

Structure:
- extract_json(): Extract JSON object from model response
- parse_yes_no(): Parse binary Yes/No answer
- parse_probability(): Parse float in [0, 1]
- parse_number(): Parse numeric answer
- parse_letter(): Parse multiple-choice letter (A/B/C/...)
- parse_instruction(): Parse proposed instruction text (eval 8)
"""

import json
import re
from typing import Optional


# Response format instruction appended to all eval prompts.
# Uses doubled braces for literal JSON in f-strings/format().
RESPONSE_FORMAT_YES_NO = (
    '\n\nRespond in JSON: {{"reasoning": "<your step by step reasoning about whether your answer will change or not>", '
    '"answer": "Yes" or "No"}}'
)

RESPONSE_FORMAT_PROBABILITY = (
    '\n\nRespond in JSON: {{"reasoning": "<your step by step reasoning about your estimation about correctness>", '
    '"answer": a number between 0 and 1}}'
)

RESPONSE_FORMAT_LETTER = (
    '\n\nRespond in JSON: {{"reasoning": "<your step-by-step reasoning>", '
    '"answer": "A", "B", or "C"}}'
)

RESPONSE_FORMAT_COMPONENT = (
    '\n\nRespond in JSON: {{"reasoning": "<your step-by-step reasoning about your choice of component>", '
    '"answer": "A", "B", or "C"}}'
)

RESPONSE_FORMAT_CODE = (
    '\n\nRespond in JSON: {{"reasoning": "<your step-by-step reasoning>", '
    '"answer": the complete code}}'
)

RESPONSE_FORMAT_ANSWER = (
    '\n\nRespond in JSON: {{"reasoning": "<your step by step reasoning about what detailed output you would give and why>", '
    '"answer": "<your predicted complete answer>"}}'
)

RESPONSE_FORMAT_INSTRUCTION = (
    '\n\nRespond with ONLY a JSON object: {{"instruction": "<your proposed instruction>"}}'
)

RESPONSE_FORMAT_CONFIDENCE = (
    '\n\nYou MUST respond with ONLY a JSON object in this exact format '
    '(no code blocks, no extra text):\n'
    '{{"answer": "<your complete answer>", "confidence": <a number between 0 and 1>}}'
)


def extract_json(response: str) -> Optional[dict]:
    """Extract a JSON object from a model response.

    Handles:
    - Plain JSON
    - JSON wrapped in markdown code blocks
    - JSON embedded in surrounding text
    - Multiple JSON objects (returns last match with known keys)

    Args:
        response: Raw model response text

    Returns:
        Parsed dict or None if no valid JSON found
    """
    if not response:
        return None

    resp = response.strip()

    # Normalize doubled braces: models sometimes output {{"key": ...}} mimicking
    # the Python format-string template syntax shown in the prompt.
    if resp.startswith("{{") and resp.endswith("}}"):
        resp = resp[1:-1]

    # Strip outer markdown code block
    if resp.startswith("```"):
        nl_idx = resp.find("\n")
        if nl_idx == -1:
            resp = resp.lstrip("`").rstrip("`").strip()
        else:
            inner = resp[nl_idx + 1 :]
            last_fence = inner.rfind("```")
            if last_fence > 0:
                resp = inner[:last_fence].strip()
            else:
                resp = inner.strip()

    # Sanitize LaTeX backslash sequences that collide with JSON escapes.
    # Models often output \boxed{...} inside JSON strings, but \b is a valid
    # JSON escape (backspace), so json.loads turns \boxed into <BS>oxed.
    # Similarly \frac → \f (form feed), \newline → \n (newline), \right → \r, \text → \t.
    # Fix: escape backslashes before known LaTeX commands that start with a
    # character that is also a JSON escape letter (b, f, n, r, t).
    def _sanitize_json_backslashes(s: str) -> str:
        # Step 1: Escape \ before LaTeX commands that collide with JSON escapes
        # \boxed, \binom, \bar, \begin → \b (backspace)
        # \frac, \forall → \f (form feed)
        # \newline → \n (newline)
        # \right, \rangle → \r (carriage return)
        # \text, \times, \to → \t (tab)
        s = re.sub(r'\\(box|bin|bar|beg|frac|for|new|rig|ran|tex|tim|to(?=[a-z]))',
                   lambda m: '\\\\' + m.group(1), s)
        # Step 2: Escape remaining lone backslashes NOT followed by valid JSON escapes
        # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        s = re.sub(r'\\(?!["\\/bfnrtu\\])', r'\\\\', s)
        return s

    sanitized = _sanitize_json_backslashes(resp)

    # Try direct parse (sanitized first, then raw).
    # Use strict=False to allow literal newlines/tabs inside JSON string values —
    # models often output JSON with raw newlines in reasoning text.
    for candidate in (sanitized, resp):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            pass

    # Search for JSON with known keys (prefer last match)
    for pattern in [
        r'\{"instruction"[^}]*\}',
        r'\{"reasoning".*?"answer"[^}]*\}',
        r'\{"answer"[^}]*\}',
    ]:
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if matches:
            raw_match = matches[-1].group()
            for candidate in (_sanitize_json_backslashes(raw_match), raw_match):
                try:
                    return json.loads(candidate, strict=False)
                except json.JSONDecodeError:
                    pass

    # Balanced brace extraction — try all { positions, prefer LAST valid JSON.
    # This handles cases where free-text before the JSON contains {} (e.g., \boxed{}).
    candidates: list[dict] = []
    pos = 0
    while pos < len(resp):
        start = resp.find("{", pos)
        if start == -1:
            break

        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(resp)):
            c = resp[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

        if end > start:
            raw_frag = resp[start : end + 1]
            for candidate in (_sanitize_json_backslashes(raw_frag), raw_frag):
                try:
                    obj = json.loads(candidate, strict=False)
                    if isinstance(obj, dict) and obj:  # non-empty dict
                        candidates.append(obj)
                        break
                except json.JSONDecodeError:
                    pass
            pos = end + 1
        else:
            pos = start + 1

    # Return last non-empty candidate (model typically puts JSON at the end)
    if candidates:
        return candidates[-1]

    return None


def parse_yes_no(response: str) -> Optional[bool]:
    """Parse a Yes/No answer from model response.

    Args:
        response: Raw model response

    Returns:
        True for Yes, False for No, None if unparseable
    """
    if not response:
        return None

    parsed = extract_json(response)
    if isinstance(parsed, dict) and "answer" in parsed:
        ans = str(parsed["answer"]).strip().lower()
        if ans in ("yes", "true") or parsed["answer"] is True:
            return True
        if ans in ("no", "false") or parsed["answer"] is False:
            return False

    resp = response.strip().lower()
    if resp.startswith("yes"):
        return True
    if resp.startswith("no"):
        return False

    # Fallback: search for "answer": "Yes"/"No" even in truncated JSON
    import re
    match = re.search(
        r'"answer"\s*:\s*"?(yes|no|true|false)"?',
        response,
        re.IGNORECASE,
    )
    if match:
        val = match.group(1).lower()
        return val in ("yes", "true")

    # Last resort: look for standalone Yes/No in the text
    if re.search(r'\byes\b', resp):
        if not re.search(r'\bno\b', resp):
            return True
    if re.search(r'\bno\b', resp):
        if not re.search(r'\byes\b', resp):
            return False

    return None


def parse_probability(response: str) -> Optional[float]:
    """Parse a probability value in [0, 1] from model response.

    Args:
        response: Raw model response

    Returns:
        Float in [0, 1] or None if unparseable
    """
    if not response:
        return None

    parsed = extract_json(response)
    if isinstance(parsed, dict) and "answer" in parsed:
        try:
            val = float(parsed["answer"])
            if 0 <= val <= 1:
                return val
        except (ValueError, TypeError):
            pass

    resp = response.strip()
    match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", resp)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 1:
            return val

    match = re.search(r"(\d+)%", resp)
    if match:
        return float(match.group(1)) / 100

    return None


def parse_number(response: str) -> Optional[float]:
    """Parse a numeric answer from model response.

    Handles JSON, \\boxed{}, and plain numbers.

    Args:
        response: Raw model response

    Returns:
        Float value or None if unparseable
    """
    if not response:
        return None

    parsed = extract_json(response)
    if isinstance(parsed, dict) and "answer" in parsed:
        try:
            return float(parsed["answer"])
        except (ValueError, TypeError):
            pass

    resp = response.strip()
    match = re.search(r"\\boxed\{([^}]+)\}", resp)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    match = re.search(r"[\d,]+\.?\d*", resp)
    if match:
        try:
            return float(match.group().replace(",", ""))
        except ValueError:
            pass

    return None


def parse_letter(response: str, valid_letters: str = "ABC") -> Optional[str]:
    """Parse a multiple-choice letter answer from model response.

    Args:
        response: Raw model response
        valid_letters: String of valid letter choices (default "ABC")

    Returns:
        Uppercase letter or None if unparseable
    """
    if not response:
        return None

    valid_set = set(valid_letters.upper())

    parsed = extract_json(response)
    if isinstance(parsed, dict) and "answer" in parsed:
        ans = str(parsed["answer"]).strip().upper()
        if ans in valid_set:
            return ans

    resp = response.strip()

    # Check if response starts with a valid letter
    for letter in valid_set:
        if resp.upper().startswith(letter) and (
            len(resp) < 3 or not resp[1].isalpha()
        ):
            return letter

    # Check last line
    last_line = resp.split("\n")[-1].strip()
    for letter in valid_set:
        if last_line in (letter, f"{letter}.", f"**{letter}**"):
            return letter

    # Regex for answer patterns
    letters_pattern = "|".join(sorted(valid_set))
    match = re.search(
        rf"(?:answer|likely|choose|select|pick|would be)[:\s]+(?:is\s+)?(?:\*\*)?({letters_pattern})(?:\*\*)?",
        resp,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    return None


def parse_instruction(response: str) -> Optional[str]:
    """Parse a proposed instruction from model response (eval 8).

    Looks for {"instruction": "..."} or {"answer": "..."} JSON format.

    Args:
        response: Raw model response

    Returns:
        Instruction string or None
    """
    if not response:
        return None

    parsed = extract_json(response)
    if parsed:
        # Prefer "instruction" key, fall back to "answer"
        text = parsed.get("instruction") or parsed.get("answer")
        if text and isinstance(text, str):
            return text.strip().strip('"')

    return None


def parse_confidence(response: str) -> tuple[Optional[str], Optional[float]]:
    """Parse answer + confidence from eval 5 phase 1 response.

    Expects: {"answer": "<answer>", "confidence": <0-1>}

    Args:
        response: Raw model response

    Returns:
        Tuple of (answer_text, confidence_float)
    """
    if not response:
        return None, None

    parsed = extract_json(response)
    if not parsed:
        return None, None

    answer = parsed.get("answer")
    confidence = None
    if "confidence" in parsed:
        try:
            conf = float(parsed["confidence"])
            if 0 <= conf <= 1:
                confidence = conf
        except (ValueError, TypeError):
            pass

    return str(answer) if answer is not None else None, confidence
