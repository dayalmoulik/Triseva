from agents.state import TriSevaState


def memory_read_node(state: TriSevaState) -> dict:
    """Load session history — handled automatically by MemorySaver."""
    return {}


def memory_write_node(state: TriSevaState) -> dict:
    """Persist session — handled automatically by MemorySaver."""
    return {}