"""A single safe calculator tool for the agentic strategy.

The model offloads arithmetic to this tool instead of doing mental math, which
is where small models tend to slip on GSM8K.
"""
from __future__ import annotations

import ast
import operator as op

_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric literals allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression like '2 + 3 * (4 - 1)'."""
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
    except Exception as e:  # report errors back to the model so it can recover
        return f"ERROR: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# OpenAI / OpenRouter tool schema for function calling.
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a single arithmetic expression and return the "
                       "exact numeric result. Use for every calculation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '48 + 48/2'.",
                }
            },
            "required": ["expression"],
        },
    },
}

DISPATCH = {"calculate": lambda args: calculate(args["expression"])}
