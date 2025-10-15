"""Utilities for Uzbek question generation fine-tuning."""

__all__ = ["DatasetConfig", "load_local_dataset"]


def __getattr__(name):
    if name in __all__:
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
