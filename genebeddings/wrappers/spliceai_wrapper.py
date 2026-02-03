"""
SpliceAI wrapper compatibility shim.

The legacy spliceai-pytorch implementation has been removed. This module now
re-exports the OpenSpliceAI wrapper to preserve import paths:

    from genebeddings.wrappers.spliceai_wrapper import SpliceAIWrapper
"""

from .openspliceai_wrapper import (
    OpenSpliceAIWrapper as SpliceAIWrapper,
    OpenSpliceAIOutput as SpliceAIOutput,
    list_available_models,
)

__all__ = ["SpliceAIWrapper", "SpliceAIOutput", "list_available_models"]
