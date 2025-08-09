"""
Pipeline stages for conversation processing.

Each stage handles a specific aspect of conversation processing:
- Input: Handle text/audio input
- Context: Build conversation context
- LLM: Interface with language model
- Service: Execute utility services
- Response: Format responses
- Memory: Persist conversation
- Output: Handle text/TTS output
"""

# Import all stages for easy access
from .input import InputHandler
from .context_builder import ContextBuilder
from .llm import LLMHandler, data_prep
from .service import ServiceDispatcher
from .response import ResponseFormatter
from .memory import MemoryManager
from .output import OutputHandler

__all__ = [
    "InputHandler",
    "ContextBuilder",
    "LLMHandler",
    "data_prep",
    "ServiceDispatcher",
    "ResponseFormatter",
    "MemoryManager",
    "OutputHandler",
]

# Stage registry for dynamic loading
STAGE_REGISTRY = {
    "input": InputHandler,
    "context": ContextBuilder,
    "llm": LLMHandler,
    "service": ServiceDispatcher,
    "response": ResponseFormatter,
    "memory": MemoryManager,
    "output": OutputHandler,
}