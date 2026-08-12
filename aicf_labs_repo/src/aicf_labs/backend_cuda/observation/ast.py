from __future__ import annotations

import re

from .model import LayerObservation


_AST_KINDS = (
    "FunctionDecl",
    "ParmVarDecl",
    "ForStmt",
    "IfStmt",
    "BinaryOperator",
    "ConditionalOperator",
    "ArraySubscriptExpr",
    "CallExpr",
    "DeclRefExpr",
)


def observe_ast_dump(ast_dump: str) -> LayerObservation:
    metrics = {
        kind: len(re.findall(rf"\b{re.escape(kind)}\b", ast_dump))
        for kind in _AST_KINDS
    }
    return LayerObservation(layer="cuda_ast", metrics=metrics)
