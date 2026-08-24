"""Small public set of callable primitive operators used by executable models."""

from .operators import AddOperator, MatMulOperator, MulOperator, ReluOperator


add = AddOperator()
matmul = MatMulOperator()
mul = MulOperator()
relu = ReluOperator()


__all__ = ("add", "matmul", "mul", "relu")
