from aicf_labs.frontend_lab import GENERIC_ELEMENTWISE_SCREEN, default_operator_registry, propagate_common_mask


def main() -> None:
    registry = default_operator_registry()
    chain = ("bias_add", "relu")
    result = propagate_common_mask(chain, registry)

    print("[operator.mask.walk]")
    for step in result.steps:
        print(
            f"{step.operator}: surviving={step.surviving_properties} "
            f"removed={step.removed_properties}"
        )

    print()
    print("[fusion.screen]")
    print(f"name={GENERIC_ELEMENTWISE_SCREEN.name}")
    print(f"candidate_survives={GENERIC_ELEMENTWISE_SCREEN.candidate_survives(result)}")


if __name__ == "__main__":
    main()
