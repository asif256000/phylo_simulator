from __future__ import annotations

from .tree_sequence_generator import TreeSequenceGenerator, TreeSequenceResult


def verify_from_config(*args, **kwargs):
    from .verify import verify_from_config as _verify_from_config

    return _verify_from_config(*args, **kwargs)


__all__ = ["TreeSequenceGenerator", "TreeSequenceResult", "verify_from_config"]
