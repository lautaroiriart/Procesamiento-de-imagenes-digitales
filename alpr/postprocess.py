import re

candidate_patterns = [
    re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"),
    re.compile(r"^[A-Z]{3}[0-9]{4}$"),
]

def fix_confusions(text: str) -> str:
    tr = str.maketrans({"O":"0", "I":"1", "B":"8"})
    return text.translate(tr)

def looks_like_plate(text: str) -> bool:
    return any(p.fullmatch(text) for p in candidate_patterns)
