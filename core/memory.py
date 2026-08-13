"""Memory Module - Converted from Notebook 05."""

from typing import List, Tuple


class SlidingWindowMemoryManager:
    """Sliding window memory retaining the last k conversation turns."""

    def __init__(self, k: int = 5):
        self.k = k
        self.history: List[Tuple[str, str]] = []

    def add_turn(self, user_query: str, ai_response: str):
        """Add a conversation turn and evict oldest if history exceeds k."""
        self.history.append((user_query, ai_response))
        if len(self.history) > self.k:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        """Format history for LLM prompt context."""
        if not self.history:
            return "None"
        return "\n".join([f"User: {u}\nAssistant: {a}" for u, a in self.history])

    def clear(self):
        """Reset conversation memory."""
        self.history.clear()
