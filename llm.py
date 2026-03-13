"""
Minimal Model base class for defense compatibility.

The defenses in defenses/ reference this class for type hints.
In practice, the object passed to defenses is always a TargetLLMAdapter
from core/utils.py, which implements the same query/batch_query interface.
"""

from typing import List, Dict, Union


class Model:
    """
    Base LLM interface used as a type hint by defense modules.
    All defense functions accept a `llm` parameter conforming to this interface.
    """

    def query(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def batch_query(
        self,
        messages_list: List[List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[str]:
        raise NotImplementedError
