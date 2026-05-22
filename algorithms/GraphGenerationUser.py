# Legacy compatibility wrapper for GraphGenerationUser
# Delegates to the fully modularized package under algorithms.graph
# Keep comments in English as requested.

from algorithms.graph.orchestrator import GraphGenerationUser

__all__ = ["GraphGenerationUser"]