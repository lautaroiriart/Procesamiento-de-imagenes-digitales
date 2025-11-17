# alpr/ml/encoding.py

# Patentes argentinas nuevas tipo AA123BB → 7 caracteres.
# Si en tu dataset hay algunas de 6 (XXX999) también entran.
MAX_LEN = 7

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_idx = {c: i for i, c in enumerate(ALPHABET)}
idx_to_char = {i: c for c, i in enumerate(ALPHABET)}

def encode_text(text: str):
    """
    Convierte la patente (ej: 'AE451UX') en una lista de índices.
    Si es más corta que MAX_LEN, rellena al final con '0' (índice 0).
    Si es más larga, la recorta.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.upper().strip()

    # recortamos y rellenamos
    text = text[:MAX_LEN].ljust(MAX_LEN, "0")

    return [char_to_idx.get(ch, 0) for ch in text]

def decode_indices(indices):
    """
    Hace el inverso: lista de índices -> string de patente.
    Elimina ceros de padding del final.
    """
    chars = [idx_to_char.get(int(i), "") for i in indices]
    plate = "".join(chars)
    # sacamos padding de '0' al final
    return plate.rstrip("0")
