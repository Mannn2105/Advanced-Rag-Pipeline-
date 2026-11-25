"""Small, dependency-free HTTP Basic authentication helper."""
import base64
import binascii
import hmac


def basic_auth_valid(header: str, expected_user: str, expected_password: str) -> bool:
    """Validate an Authorization: Basic header using constant-time comparisons."""
    if not header or not expected_user or not expected_password:
        return False
    scheme, separator, encoded = header.partition(" ")
    if not separator or scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    user_ok = hmac.compare_digest(username, expected_user)
    password_ok = hmac.compare_digest(password, expected_password)
    return user_ok and password_ok
