import warnings
warnings.filterwarnings("ignore")

import ast
import operator
from langchain_core.tools import tool

# Safe operators only
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
    """Recursively evaluate an AST node safely."""
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
    """
    Safely evaluate a mathematical expression.
    Use this for eligibility calculations, income thresholds,
    unit conversions, or any arithmetic needed to answer a question.
    Examples: '6000 / 3', '14200 * 12', '2 ** 8'
    """
    try:
        expression = expression.strip()
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"