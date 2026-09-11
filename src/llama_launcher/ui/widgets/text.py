"""Text helpers for Qt widget strings."""


def literal_ampersands(text: str) -> str:
    """`text` with every lone "&" doubled: a Qt widget title or button label
    reads a lone "&" as a mnemonic marker, and "&&" renders one ampersand."""
    return text.replace("&", "&&")
