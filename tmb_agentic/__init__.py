"""TMB Agentic: deterministic worlds and a mechanical verdict for tool-using agents."""

__version__ = "0.1.0"

from .engine import World, ToolResult, FaultEngine, AssertionEngine  # noqa: E402,F401
from .worlds import FsWorld, DbWorld, ApiWorld, make_world  # noqa: E402,F401
