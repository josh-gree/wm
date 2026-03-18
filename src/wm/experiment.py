from pydantic import BaseModel


class Experiment:
    name: str
    gpu: str | None = None
    timeout: int | None = None
    ephemeral_disk: int | None = None

    Config: type[BaseModel]

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
