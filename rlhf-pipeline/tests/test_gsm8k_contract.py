"""
Comprehensive Test Suite for GSM8K Narrow Answer-Checking Contract
===================================================================
Tests all edge cases specified in the user directive:
1. Correct answers (integers, decimals, thousands separators, currency)
2. Incorrect answers
3. Negative numbers (negative vs positive, formatted signs)
4. Fractions (LaTeX, standard fractions, reduction, division by zero)
5. Percentages (50% = 0.5, 50% != 50)
6. Units scrutiny & misleading extra text (ambiguity rejection, multiple numbers)
7. Malformed tags & format compliance
"""

import pytest
from rlhf_pipeline.rewards.gsm8k_contract import (
    extract_answer_content,
    normalize_numeric_string,
    verify_gsm8k_accuracy,
    verify_format_compliance,
)


def test_correct_integer_and_decimal_answers():
    # Standard clean answers
    assert verify_gsm8k_accuracy("<think>...</think><answer>42</answer>", "42")
    assert verify_gsm8k_accuracy("<think>...</think><answer>42.0</answer>", "42")
    assert verify_gsm8k_accuracy("<think>...</think><answer>42</answer>", "42.00")
    assert verify_gsm8k_accuracy("<think>...</think><answer>3.14</answer>", "3.14")
    assert verify_gsm8k_accuracy("<think>...</think><answer>0.5</answer>", ".5")


def test_thousands_separators():
    # Valid thousands grouping
    assert verify_gsm8k_accuracy("<think>...</think><answer>1,000</answer>", "1000")
    assert verify_gsm8k_accuracy("<think>...</think><answer>1,234,567</answer>", "1234567")
    assert verify_gsm8k_accuracy("<think>...</think><answer>448,000</answer>", "448000")

    # Malformed comma placement must be rejected
    assert not verify_gsm8k_accuracy("<think>...</think><answer>1,2,3</answer>", "123")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>1,00</answer>", "100")


def test_currency_formatting():
    assert verify_gsm8k_accuracy("<think>...</think><answer>$100</answer>", "100")
    assert verify_gsm8k_accuracy("<think>...</think><answer>-$50</answer>", "-50")
    assert verify_gsm8k_accuracy("<think>...</think><answer>50$</answer>", "50")


def test_incorrect_answers():
    assert not verify_gsm8k_accuracy("<think>...</think><answer>41</answer>", "42")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>100</answer>", "200")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>0</answer>", "1")


def test_negative_numbers():
    assert verify_gsm8k_accuracy("<think>...</think><answer>-15</answer>", "-15")
    assert verify_gsm8k_accuracy("<think>...</think><answer>- 15</answer>", "-15")
    assert verify_gsm8k_accuracy("<think>...</think><answer>-$15</answer>", "-15")
    
    # Negative vs positive must NOT match
    assert not verify_gsm8k_accuracy("<think>...</think><answer>15</answer>", "-15")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>-15</answer>", "15")


def test_fractions():
    # Standard and LaTeX fractions
    assert verify_gsm8k_accuracy("<think>...</think><answer>3/4</answer>", "0.75")
    assert verify_gsm8k_accuracy(r"<think>...</think><answer>\frac{3}{4}</answer>", "0.75")
    assert verify_gsm8k_accuracy(r"<think>...</think><answer>\frac{1}{2}</answer>", "0.5")
    assert verify_gsm8k_accuracy("<think>...</think><answer>2/4</answer>", "1/2")

    # Negative fraction
    assert verify_gsm8k_accuracy("<think>...</think><answer>-1/2</answer>", "-0.5")

    # Division by zero must be rejected
    assert not verify_gsm8k_accuracy("<think>...</think><answer>5/0</answer>", "5")


def test_percentages_scrutiny():
    # 50% means 0.5, NOT 50
    assert verify_gsm8k_accuracy("<think>...</think><answer>50%</answer>", "0.5")
    assert verify_gsm8k_accuracy("<think>...</think><answer>100%</answer>", "1.0")
    assert verify_gsm8k_accuracy("<think>...</think><answer>42%</answer>", "0.42")

    # Critical check: 50% must NOT match 50!
    assert not verify_gsm8k_accuracy("<think>...</think><answer>50%</answer>", "50")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>50</answer>", "50%")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>100%</answer>", "100")


def test_misleading_extra_text_and_ambiguity():
    # Single clean number in text is allowed
    assert verify_gsm8k_accuracy("<think>...</think><answer>The final answer is 42</answer>", "42")
    assert verify_gsm8k_accuracy("<think>...</think><answer>#### 42</answer>", "42")
    assert verify_gsm8k_accuracy("<think>...</think><answer>x = 42</answer>", "42")

    # Ambiguous multi-number or alternative outputs MUST be rejected
    assert not verify_gsm8k_accuracy("<think>...</think><answer>42 or 43</answer>", "42")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>between 10 and 20</answer>", "15")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>either 5 or 10</answer>", "5")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>He had 50 kg of apples, giving 500 dollars</answer>", "50")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>50 kg of apples</answer>", "500")

    # Attached unit strings without whitespace (e.g. 50kg) must be rejected
    assert not verify_gsm8k_accuracy("<think>...</think><answer>50kg</answer>", "50")


def test_malformed_tags_and_non_numeric_outputs():
    # Empty answer
    assert not verify_gsm8k_accuracy("<think>...</think><answer></answer>", "42")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>   </answer>", "42")
    assert not verify_gsm8k_accuracy("", "42")

    # Multiple or unclosed answer tags
    assert not verify_gsm8k_accuracy("<think>...</think><answer>42<answer>43</answer>", "42")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>42</answer><answer>43</answer>", "42")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>42", "42")

    # Non-numeric text
    assert not verify_gsm8k_accuracy("<think>...</think><answer>I don't know</answer>", "42")
    assert not verify_gsm8k_accuracy("<think>...</think><answer>No solution</answer>", "0")
    assert not verify_gsm8k_accuracy("There is no solution to this equation.", "42")


def test_format_compliance():
    # Valid order
    assert verify_format_compliance("<think>Step 1</think><answer>42</answer>")
    assert verify_format_compliance("<think>\nStep 1\nStep 2\n</think>\n<answer>42</answer>")

    # Invalid order or missing tags
    assert not verify_format_compliance("<answer>42</answer><think>Step 1</think>")
    assert not verify_format_compliance("<answer>42</answer>")
    assert not verify_format_compliance("<think>Step 1</think>")
    assert not verify_format_compliance("Plain text")
