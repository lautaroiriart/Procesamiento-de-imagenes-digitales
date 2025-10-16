
import re
PATTERNS = [
    re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"),
    re.compile(r"^[A-Z]{3}[0-9]{4}$"),
]
def fix_confusions(text: str) -> str:
    return text.translate(str.maketrans({"O":"0","I":"1","B":"8"}))
def looks_like_plate(t: str) -> bool:
    return any(p.fullmatch(t) for p in PATTERNS)
