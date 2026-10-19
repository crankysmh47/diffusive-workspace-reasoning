"""Neural model components, imported only when this package is requested."""

from .workspace import FixedWorkspace, WorkspaceDelta
from .interface import SetCrossAttentionAdapter

__all__ = ["FixedWorkspace", "WorkspaceDelta", "SetCrossAttentionAdapter"]
