from typing import List, Union, Any
from lm_eval.utils import handle_arg_string as handle_arg_string_2
from lm_eval.api.metrics import handle_arg_string, handle_arg_string_math_verify

import ast
import logging
import math
import random
import re
import string
from collections.abc import Iterable

import numpy as np
import sacrebleu
import math_verify

from lm_eval.api.registry import register_aggregation, register_metric
eval_logger = logging.getLogger(__name__)


def doc_to_target(doc) -> List[Union[str, float]]:
    return [
        handle_arg_string_2(a) for a in doc["answer"]
    ]

def _extract_list_string(s: str) -> str:
    """
    Extracts the first shallow bracketed list from the input string.
    Returns the substring including brackets, or an empty string if none found.
    """
    m = re.search(r'\[[^\[\]]*\]', s)
    return m.group(0) if m else ''

def list_match(
    predictions: Union[str, List[str]],
    references: Union[List[Any], List[List[Any]]],
    ignore_case: bool = False,
    use_math_verify: bool = False,
    ):
    predictions = _extract_list_string(predictions)
    # Map a simple sample
    if not isinstance(predictions, list):
        predictions = [predictions]
        references = [references]

    scores: List[int] = []
    for pred, ref in zip(predictions, references):

        if ignore_case:
            pred = pred.lower()
            ref = [r.lower() if isinstance(r, str) else r for r in ref]
        
        if use_math_verify:
            ref = [handle_arg_string_math_verify(r) for r in ref]
        ref = set(ref)
        try:
            pred = ast.literal_eval(pred)
            assert isinstance(pred, list)
            if use_math_verify:
                pred = [handle_arg_string_math_verify(p) for p in pred]
            else:
                pred = [handle_arg_string(str(p)) for p in pred]
            pred = set(pred)
        except (AssertionError, SyntaxError, ValueError):
            eval_logger.debug(
                f"Model did not produces a parsable array. Casting to an empty result.\nModel output: '{pred}'"
            )
            pred = set()

        scores.append(int(
            len(pred) == len(ref) == len(pred.intersection(ref))
        ))
    if use_math_verify:
        return {"list_match_math_verify": np.mean(scores)}
    else:
        return {"list_match": np.mean(scores)}


def list_match_math_verify(
    predictions: Union[str, List[str]],
    references: Union[List[Any], List[List[Any]]],
    ignore_case: bool = False,
    use_math_verify: bool = True,
    ):
    predictions = _extract_list_string(predictions)
    # Map a simple sample
    if not isinstance(predictions, list):
        predictions = [predictions]
        references = [references]

    scores: List[int] = []
    for pred, ref in zip(predictions, references):

        if ignore_case:
            pred = pred.lower()
            ref = [r.lower() if isinstance(r, str) else r for r in ref]
        
        if use_math_verify:
            ref = [handle_arg_string_math_verify(r) for r in ref]
        ref = set(ref)
        try:
            pred = ast.literal_eval(pred)
            assert isinstance(pred, list)
            if use_math_verify:
                pred = [handle_arg_string_math_verify(p) for p in pred]
            else:
                pred = [handle_arg_string(str(p)) for p in pred]
            pred = set(pred)
        except (AssertionError, SyntaxError, ValueError):
            eval_logger.debug(
                f"Model did not produces a parsable array. Casting to an empty result.\nModel output: '{pred}'"
            )
            pred = set()

        scores.append(int(
            len(pred) == len(ref) == len(pred.intersection(ref))
        ))
    if use_math_verify:
        return {"list_match_math_verify": np.mean(scores)}
    else:
        return {"list_match": np.mean(scores)}