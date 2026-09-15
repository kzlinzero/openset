"""终端展示层：只负责打印，不包含任何模型 / 张量逻辑。"""

from __future__ import annotations


class C:
    """ANSI 颜色 / 样式常量。"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


WIDTH = 60


def title(text: str, color: str = C.BRIGHT_CYAN) -> None:
    """居中标题块。"""
    line = "═" * WIDTH
    print(f"\n{C.BOLD}{color}{line}{C.RESET}")
    print(f"{C.BOLD}{color}  {text}{C.RESET}")
    print(f"{C.BOLD}{color}{line}{C.RESET}")


def section(text: str, color: str = C.BRIGHT_YELLOW) -> None:
    """子标题。"""
    line = "─" * WIDTH
    print(f"\n{C.BOLD}{color}{line}{C.RESET}")
    print(f"{C.BOLD}{color}▶ {text}{C.RESET}")
    print(f"{C.BOLD}{color}{line}{C.RESET}")


def kv(key: str, value, key_color: str = C.CYAN, val_color: str = C.WHITE) -> None:
    """对齐的 key/value 行。"""
    print(f"  {key_color}{key:<12}{C.RESET}{val_color}{value}{C.RESET}")


def ok(text: str) -> None:
    print(f"  {C.GREEN}✓ {text}{C.RESET}")


def token_line(idx: int, text: str, token_id: int, vec, tag_color: str) -> None:
    """打印单个 token 的文本、ID 和前 5 维向量。"""
    display = text.replace("\n", "\\n").replace("\t", "\\t")
    print(
        f"  {tag_color}位置 {idx:>3}{C.RESET}  "
        f"{C.BOLD}{C.WHITE}'{display}'{C.RESET}  "
        f"{C.DIM}(ID: {token_id}){C.RESET}"
    )
    vals = "  ".join(f"{v:+.4f}" for v in vec)
    print(f"           {C.MAGENTA}向量前5值:{C.RESET} [{vals}]")


def banner(text: str = "✓ 全部完成") -> None:
    line = "═" * WIDTH
    print(f"\n{C.BOLD}{C.BRIGHT_GREEN}{line}{C.RESET}")
    print(f"{C.BOLD}{C.BRIGHT_GREEN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.BRIGHT_GREEN}{line}{C.RESET}\n")