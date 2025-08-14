import re
import math_verify
from typing import List


def doc_to_text(doc):
    """Extract the question and context from the document."""
    # Get the question
    question = doc.get("question", "").strip()
    
    # Get the context paragraphs
    paragraphs = doc.get("paragraphs", [])
    context = "\n\n".join(paragraphs)
    
    # Format the prompt
    prompt = f"{context}\n\n {question}\n\n"
    
    return prompt


def doc_to_target(doc):
    """Extract the expected answer from the document."""
    # Get the ground truth answer
    return str(doc.get("ground_truth", "")).strip()


def extract_answer_from_text(text):
    """Extract the final answer from generated text."""
    # Handle boolean/yes-no answers first
    text_lower = text.lower()
    
    # Check for explicit yes/no, true/false patterns
    bool_patterns = [
        r"(?:the answer is:?\s*)(yes|no|true|false)",
        r"(?:therefore,?\s*)(yes|no|true|false)",
        r"(?:so,?\s*)(yes|no|true|false)",
        r"^(yes|no|true|false)[.,!?]?\s*$",
    ]
    
    for pattern in bool_patterns:
        match = re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE)
        if match:
            answer = match.group(1).strip().title()
            return answer
    
    # Look for calculation results
    calc_patterns = [
        r"(?:=|equals?)\s*(-?[0-9]+(?:\.[0-9]+)?)\s*(?:\n|$|[^\d])",
        r"(?:approximately|about|roughly)\s*(\$?-?[0-9]+(?:[.,][0-9]+)*(?:\s*(?:million|thousand|k|m|b))?%?)",
        r"(?:which gives us|resulting in|equals?)\s*:?\s*(\$?-?[0-9]+(?:[.,][0-9]+)*(?:\s*(?:million|thousand|k|m|b))?%?)",
    ]
    
    for pattern in calc_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
    
    # Standard answer patterns
    patterns = [
        r"### Answer:\s*([^\n]+)",
        r"Answer:\s*([^\n]+)",
        r"Final answer:\s*([^\n]+)",
        r"The answer is:?\s*([^\n]+)",
        r"Therefore,?\s*([^\n]+)",
        r"So,?\s*([^\n]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).strip()
            # Clean up trailing punctuation
            candidate = re.sub(r'[.!?,;:]+$', '', candidate)
            
            # If it contains a number, extract it
            if re.search(r'\d', candidate):
                num_match = re.search(r'(\$?-?[0-9]+(?:[.,][0-9]+)*(?:\s*(?:million|thousand|k|m|b))?%?)', candidate)
                if num_match:
                    return num_match.group(1).strip()
            
            return candidate
    
    # Look for standalone numbers at the end of lines
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        line = re.sub(r'[.!?,;:]+$', '', line)
        number_match = re.search(r'(\$?-?[0-9]+(?:[.,][0-9]+)*(?:\s*(?:million|thousand|k|m|b))?%?)\s*$', line)
        if number_match:
            return number_match.group(1).strip()
    
    # Last resort - find any number in the text
    all_numbers = re.findall(r'(\$?-?[0-9]+(?:[.,][0-9]+)*(?:\s*(?:million|thousand|k|m|b))?%?)', text)
    if all_numbers:
        return all_numbers[-1].strip()
    
    # Return the last non-empty line if no number found
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    
    return text.strip()


def normalize_answer(answer):
    """Normalize an answer for comparison."""
    answer_str = str(answer).strip()
    
    # Handle percentage conversion
    if answer_str.endswith('%'):
        try:
            num_str = answer_str.rstrip('%').strip()
            return str(float(num_str) / 100.0)
        except ValueError:
            pass
    
    # Handle currency
    currency_prefix = ""
    if answer_str.startswith('$'):
        currency_prefix = "$"
        answer_str = answer_str.lstrip('$').strip()
    
    # Handle suffixes
    suffixes = {
        "million": 1e6,
        "thousand": 1e3,
        "k": 1e3,
        "m": 1e6,
        "b": 1e9,
    }
    
    for suffix, multiplier in suffixes.items():
        pattern = rf"^(-?[0-9]+(?:\.[0-9]+)?)\s*{suffix}$"
        match = re.match(pattern, answer_str.lower())
        if match:
            try:
                num_part = match.group(1)
                return str(float(num_part) * multiplier)
            except ValueError:
                pass
    
    # Remove commas from numbers
    if re.match(r'^-?[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?$', answer_str):
        answer_str = answer_str.replace(',', '')
    
    return answer_str


def equal_value_custom(predictions: List[str], references: List[str]) -> dict:
    """Check if prediction equals reference with custom normalization."""
    assert len(predictions) == len(references) == 1
    
    # Extract answer from generated text
    pred_text = predictions[0]
    pred_answer = extract_answer_from_text(pred_text)
    
    # Get reference answer
    ref_answer = references[0]
    
    # Normalize both answers
    pred_norm = normalize_answer(pred_answer)
    ref_norm = normalize_answer(ref_answer)
    
    # Check for boolean equality
    bool_map = {
        "yes": "Yes", "no": "No", 
        "true": "True", "false": "False",
    }
    
    pred_lower = pred_norm.lower()
    ref_lower = ref_norm.lower()
    
    if pred_lower in bool_map or ref_lower in bool_map:
        pred_bool = bool_map.get(pred_lower, pred_norm)
        ref_bool = bool_map.get(ref_lower, ref_norm)
        return int(pred_bool == ref_bool)
    
    # Try numerical comparison
    try:
        pred_float = float(pred_norm)
        ref_float = float(ref_norm)
        
        # Check if integers
        if pred_float == int(pred_float) and ref_float == int(ref_float):
            return int(int(pred_float) == int(ref_float))
        
        # Use tolerance for floats
        if abs(ref_float) < 1e-10:
            return int(abs(pred_float - ref_float) < 1e-6)
        else:
            relative_error = abs(pred_float - ref_float) / abs(ref_float)
            return int(relative_error < 1e-6)
    except (ValueError, TypeError):
        # Fall back to string comparison
        return int(pred_norm == ref_norm)


def equal_math_verify(predictions: List[str], references: List[str]) -> dict:
    """Use math_verify to check answer equality."""
    assert len(predictions) == len(references) == 1
    
    # Extract answer from generated text
    pred_text = predictions[0]
    pred_answer = extract_answer_from_text(pred_text)
    
    # Get reference answer
    ref_answer = references[0]
    
    try:
        # Try parsing both with math_verify
        pred_result = math_verify.parse(pred_answer)
        ref_result = math_verify.parse(ref_answer)
        
        if pred_result and ref_result:
            # Use math_verify's verify function
            return int(math_verify.verify(ref_answer, pred_answer))
    except:
        pass
    
    # Fall back to custom comparison
    return equal_value_custom(predictions, references) 