"""
TriSeva Safe AST Mathematical Calculator Tool.

Provides secure, AST-parsed arithmetic expression evaluation for citizen land conversions,
income threshold calculations, and policy eligibility verification without eval() vulnerabilities.
"""

import warnings
warnings.filterwarnings("ignore")

import ast
import operator
from langchain_core.tools import tool

# Whitelisted safe mathematical AST operators
SAFE_OPERATORS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Mod:  operator.mod,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """Recursively evaluates an AST node using whitelisted safe arithmetic operators.

    Args:
        node (ast.AST): AST node instance.

    Returns:
        int | float: Calculated numerical result.

    Raises:
        ValueError: If unsupported AST nodes or operators are encountered.
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


@tool
def calculator_tool(expression: str) -> str:
    """Safely evaluates a mathematical expression string.

    Use for scheme income eligibility limits, land area conversions, dosage totals, or arithmetic.
    Examples: '6000 / 3', '14200 * 12', '1.5 * 2.47'.

    Args:
        expression (str): String containing arithmetic expression.

    Returns:
        str: Formatted calculation result string (e.g. '6000 / 3 = 2000.0').
    """
    try:
        expression = expression.strip()
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"