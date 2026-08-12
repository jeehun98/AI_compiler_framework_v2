# Frontend Operator Marking

The frontend begins with two explicit structures.

## 1. Mask vocabulary

`frontend_lab/marking/mask.py` owns global bit meanings. Each bit is stable within the experiment format and belongs to a semantic domain such as algebra, tensor dependency, effects, or transform screening.

A set bit means the operator explicitly advertises that property. The current model intentionally uses binary data because the first purpose is cheap screening and propagation.

## 2. Per-operator marking

`frontend_lab/marking/builtin.py` assigns masks to individual mathematical operators.

The marks do not encode a full proof system. In particular, floating-point algebra may require a numerical contract even when a mathematical operator is marked associative or commutative.

## Propagation

For an operator chain, `propagate_common_mask()` performs bit intersection and records when each property disappears.

```text
mask(op0)
   AND mask(op1)
   AND mask(op2)
   ...
```

A fusion screen then checks whether its required bits are still present. Surviving bits mean that this cheap filter has **not eliminated** the candidate. Use-def, shape, numerical, and semantic legality remain later analyses.
