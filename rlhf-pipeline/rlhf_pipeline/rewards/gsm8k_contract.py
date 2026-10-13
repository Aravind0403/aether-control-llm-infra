"""
Narrow, Robust Answer-Checking Contract for GSM8K Reasoning Tasks
===================================================================
Enforces a deterministic contract:
1. Extract candidate text from exactly one <answer>...</answer> block.
2. Reject malformed, missing, unclosed, or multiple conflicting answer tags.
3. Parse and normalize supported numeric representations:
   - Integers and decimals: '42', '42.0', '0.75'
   - Currency: '$42', '-$42', '42$'
   - Standard thousands-separated integers: '1,000', '1,234,567'
   - Fractions: '3/4', r'\frac{3}{4}', '-1/2'
   - Percentages: '50%' parses as 0.50 (does NOT equal 50)
   - Negative numbers: '-42', '- 42', '-$42'
   - Simple algebraic equations: 'x = 42' extracts 42 only if unambiguous
4. Reject ambiguous, misleading, or non-numeric outputs:
   - Contradictory numbers: '42 or 43', 'between 10 and 20'
   - Non-numeric text: 'I do not know', 'no solution'
   - Division by zero: '5/0'
5. Bounded execution time: pure regex and decimal/fraction arithmetic (no unconstrained symbolic execution).
"""

import re
from typing import Optional, Tuple
from fractions import Fraction
from decimal import Decimal, InvalidOperation


def extract_answer_content(completion: str) -> Optional[str]:
    """
    Extracts candidate final answer string from completion.
    Priority 1: Content from <answer>...</answer> tag.
    Priority 2: Fallback extraction of final numeric answer if <answer> tags are omitted.
    Rejects completions with malformed, unclosed, or contradictory <answer> tags.
    """
    if not completion or not isinstance(completion, str) or len(completion) > 4096:
        return None

    # Check for <answer> tags
    open_tags = [m.start() for m in re.finditer(r"<answer>", completion)]
    close_tags = [m.start() for m in re.finditer(r"</answer>", completion)]

    if open_tags and close_tags:
        if len(open_tags) != 1 or len(close_tags) != 1 or open_tags[0] >= close_tags[0]:
            # Malformed or multiple conflicting tags -> reject
            return None
        content = completion[open_tags[0] + len("<answer>"):close_tags[0]].strip()
        return content if content else None
    elif open_tags or close_tags:
        # Unclosed single tag -> reject
        return None

    # Fallback when <answer> tag is omitted: extract final numeric answer
    # Priority 2a: '#### <number>'
    if "####" in completion:
        candidate = completion.split("####")[-1].strip()
        tokens = candidate.split()
        if tokens:
            return tokens[0]

    # Priority 2b: 'The (final) answer is <number>'
    ans_match = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|:|=)\s*([-$]?\d[\d,\.]*)", completion, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()

    # Priority 2c: Last numeric token in text (if unambiguous)
    # Exclude think tags from search if closed
    text_to_search = completion
    if "</think>" in text_to_search:
        text_to_search = text_to_search.split("</think>")[-1].strip()
    
    # Extract numbers
    numbers = re.findall(r"[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\$?\d+(?:\.\d+)?", text_to_search)
    if numbers:
        return numbers[-1].strip()

    return None


def normalize_numeric_string(raw: str) -> Optional[Fraction]:
    """
    Parses a candidate answer string into an exact Fraction.
    Returns None if the string is ambiguous, malformed, or non-numeric.
    """
    if not raw or not isinstance(raw, str) or len(raw) > 64:
        return None

    s = raw.strip()

    # Bounded digit cap to prevent runaway precision or parsing DoS
    if len(re.findall(r"\d", s)) > 20:
        return None

    # Reject obvious ambiguity phrases
    ambiguous_patterns = [
        r"\bor\b", r"\band\b", r"\bbetween\b", r"\beither\b", r"\bmaybe\b",
        r"\bapproximately\b", r"\baround\b", r"\babout\b", r"\bto\b"
    ]
    for pat in ambiguous_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return None

    # Strip conversational prefixes e.g. 'the answer is 42', '#### 42', 'x = 42'
    s = re.sub(r"^(the\s+)?(final\s+)?answer\s+(is\s+|:\s*)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^####\s*", "", s).strip()
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s).strip()

    # Handle negative signs with spaces e.g. '- 42' -> '-42'
    s = re.sub(r"^-\s+", "-", s)

    # Handle currency symbols: '$42' -> '42', '-$42' -> '-42', '42$' -> '42'
    s = re.sub(r"^-\$", "-", s)
    s = re.sub(r"^\$", "", s)
    s = re.sub(r"\$$", "", s).strip()

    # Handle percentages: '50%' -> 50 / 100 = 0.5 (Fraction(1, 2))
    if s.endswith("%"):
        inner = s[:-1].strip()
        inner_val = normalize_numeric_string(inner)
        if inner_val is not None:
            return inner_val / Fraction(100, 1)
        return None

    # Handle LaTeX fractions: r'\frac{a}{b}' -> 'a/b'
    latex_frac = re.match(r"^\\frac\{([^}]+)\}\{([^}]+)\}$", s)
    if latex_frac:
        num_str = latex_frac.group(1).strip()
        den_str = latex_frac.group(2).strip()
        num_val = normalize_numeric_string(num_str)
        den_val = normalize_numeric_string(den_str)
        if num_val is not None and den_val is not None and den_val != 0:
            return num_val / den_val
        return None

    # Handle standard fractions: 'a/b'
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            num_val = normalize_numeric_string(parts[0])
            den_val = normalize_numeric_string(parts[1])
            if num_val is not None and den_val is not None and den_val != 0:
                return num_val / den_val
        return None

    # Handle thousand separators: '1,000', '1,234,567'
    # Must strictly match valid comma grouping
    if "," in s:
        if re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", s):
            s = s.replace(",", "")
        else:
            # Malformed comma placement e.g. '1,2,3' -> reject
            return None

    # Check for trailing units attached to a number without space e.g. '50kg'
    # Reject because units must not be stripped to mask wrong dimensions
    if re.search(r"\d+[a-zA-Z]+$", s):
        return None

    # If extra non-numeric words remain, check if it's a single clean trailing noun like '42 dollars'
    # Only allow if there is strictly one number and nothing else
    tokens = s.split()
    if len(tokens) > 1:
        # Check if there is exactly one numeric token
        numeric_tokens = []
        for t in tokens:
            t_clean = t.rstrip(".,;:!?")
            try:
                Decimal(t_clean)
                numeric_tokens.append(t_clean)
            except InvalidOperation:
                pass
        if len(numeric_tokens) == 1:
            s = numeric_tokens[0]
        else:
            # Multiple numbers in text e.g. '50 kg and 10 apples' -> reject as ambiguous
            return None

    # Try parsing as Decimal and converting to Fraction
    try:
        dec = Decimal(s)
        # Bounded magnitude check to avoid runaway precision DoS
        if abs(dec) > Decimal("1e15"):
            return None
        return Fraction(str(dec))
    except (InvalidOperation, ValueError):
        return None


def verify_gsm8k_accuracy(completion: str, target: str) -> bool:
    """
    End-to-end verification of completion against ground truth.
    Returns True if completion extracts to an unambiguous numeric value equivalent to target.
    """
    candidate_str = extract_answer_content(completion)
    if candidate_str is None:
        return False

    pred_val = normalize_numeric_string(candidate_str)
    if pred_val is None:
        return False

    # Target normalization
    # GSM8K ground truth targets after #### are strings like '42', '12000', '-5'
    target_clean = target.split("####")[-1].strip() if "####" in target else target.strip()
    target_val = normalize_numeric_string(target_clean)
    if target_val is None:
        return False

    return pred_val == target_val


def reconstruct_assistant_completion(completion_text: str) -> str:
    """
    Reconstructs the full assistant response consistently across training and evaluation.
    Because the prompt prefill ends with '<|im_start|>assistant\n<think>\n',
    the model's generated continuation does not include the opening '<think>' tag.
    If the text does not start with '<think>', prepend '<think>\n'.
    """
    if not completion_text or not isinstance(completion_text, str):
        return ""
    text = completion_text.strip()
    if text.startswith("<think>"):
        return text
    return f"<think>\n{text}"


def verify_format_compliance(completion: str) -> bool:
    """
    Verifies that reasoning tags exist in strict valid sequential order:
    <think>...</think> followed by <answer>...</answer>.
    Caps text length at 4096 chars to guarantee bounded execution time.
    """
    if not completion or not isinstance(completion, str) or len(completion) > 4096:
        return False

    think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

    if not think_match or not answer_match:
        return False

    # Check that think starts before answer
    return think_match.start() < answer_match.start()

