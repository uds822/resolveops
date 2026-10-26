import hashlib
import json


def make_stable_id(namespace: str, *parts: str) -> str:
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")

    namespace = namespace.strip()

    if not namespace:
        raise ValueError("namespace cannot be empty")

    if not parts:
        raise ValueError("at least one identity part is required")

    if not all(isinstance(part, str) for part in parts):
        raise TypeError("all identity parts must be strings")

    payload = json.dumps(
        [namespace, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return f"{namespace}_{digest}"
