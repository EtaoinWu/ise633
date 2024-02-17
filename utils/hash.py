import hashlib


def my_hash(x: str) -> int:
    return int(hashlib.sha256(x.encode("utf-8")).hexdigest(), 16) % 10**9
