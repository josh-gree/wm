import typing
from dataclasses import fields


def _coerce(value: str, target_type: type):
    if target_type is bool:
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot convert {value!r} to bool")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def resolve_config(hyper_params_cls: type, overrides: dict[str, str]) -> dict:
    field_map = {f.name: f for f in fields(hyper_params_cls)}

    unknown = set(overrides.keys()) - set(field_map.keys())
    if unknown:
        raise ValueError(f"Unknown hyperparameter(s): {', '.join(sorted(unknown))}")

    try:
        type_hints = typing.get_type_hints(hyper_params_cls)
    except Exception:
        type_hints = {f.name: f.type for f in fields(hyper_params_cls)}

    # Start with defaults
    defaults = hyper_params_cls()
    config = {f.name: getattr(defaults, f.name) for f in fields(hyper_params_cls)}

    # Apply overrides with type coercion
    for key, value in overrides.items():
        target_type = type_hints[key]
        # Handle string type annotations that weren't resolved
        if isinstance(target_type, str):
            type_map = {"int": int, "float": float, "bool": bool, "str": str}
            target_type = type_map.get(target_type, str)
        config[key] = _coerce(value, target_type)

    return config
