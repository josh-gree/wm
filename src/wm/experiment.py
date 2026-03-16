from pydantic import BaseModel


class Experiment:
    name: str
    gpu: str | None = None
    timeout: int | None = None
    extra_dependencies: list[str] | None = None
    apt_packages: list[str] | None = None
    volume: str | None = None
    data_mount: str | None = None
    dockerfile: str | None = None

    Config: type[BaseModel]

    _CONTAINER_FIELDS = (
        "gpu",
        "timeout",
        "extra_dependencies",
        "apt_packages",
        "volume",
        "data_mount",
        "dockerfile",
    )

    @classmethod
    def container_dict(cls) -> dict | None:
        d = {}
        for field in cls._CONTAINER_FIELDS:
            val = getattr(cls, field, None)
            if val is not None:
                d[field] = val
        return d or None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name") or not isinstance(cls.name, str):
            raise TypeError(
                f"Experiment subclass {cls.__name__} must define a 'name' string attribute"
            )
        if not hasattr(cls, "Config") or not (
            isinstance(cls.Config, type) and issubclass(cls.Config, BaseModel)
        ):
            raise TypeError(
                f"Experiment subclass {cls.__name__} must define a 'Config' class inheriting from BaseModel"
            )
        if "run" not in cls.__dict__:
            raise TypeError(
                f"Experiment subclass {cls.__name__} must define a 'run' static method"
            )
