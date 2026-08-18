"""A minimal, self-contained tool so the CLI has something to run out of the
box without pulling in the benchmark's larger tool set.
"""

from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from .tools import Tool

_OPS: dict[type[ast.AST], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        return _OPS[type(node.op)](left, right)  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))  # type: ignore[operator]
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="An arithmetic expression, e.g. '(12 * 7) + 3'")


def _calculate(args: CalculatorArgs) -> str:
    tree = ast.parse(args.expression, mode="eval")
    return str(_safe_eval(tree.body))


def calculator_tool() -> Tool[CalculatorArgs]:
    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression using + - * / ** and parentheses.",
        parameters=CalculatorArgs,
        func=_calculate,
    )
