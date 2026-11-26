"""Small, dependency-free helpers for shared HTTP Basic authentication."""
from __future__ import annotations

import base64
import binascii
import secrets


def is_basic_auth_valid(
    authorization_header: str | None,
    expected_username: str,
    expected_password: str,
) -> bool:
    """Validate a Basic Authorization header without leaking credential values."""
    if not expected_username or not expected_password or ":" in expected_username:
        return False
    if not authorization_header:
        return False

    scheme, separator, token = authorization_header.partition(" ")
    if not separator or scheme.lower() != "basic" or not token.strip():
        return False

    try:
        decoded = base64.b64decode(token.strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False

    username, separator, password = decoded.partition(":")
    if not separator:
        return False

    username_matches = secrets.compare_digest(username, expected_username)
    password_matches = secrets.compare_digest(password, expected_password)
    return username_matches and password_matches
