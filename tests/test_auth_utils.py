import base64

from auth_utils import is_basic_auth_valid


def basic_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def test_accepts_valid_basic_credentials():
    assert is_basic_auth_valid(
        basic_header("demo-user", "correct-horse"),
        "demo-user",
        "correct-horse",
    )


def test_accepts_colons_inside_password():
    assert is_basic_auth_valid(
        basic_header("demo-user", "part-one:part-two"),
        "demo-user",
        "part-one:part-two",
    )


def test_rejects_wrong_password():
    assert not is_basic_auth_valid(
        basic_header("demo-user", "wrong"),
        "demo-user",
        "correct-horse",
    )


def test_rejects_wrong_username():
    assert not is_basic_auth_valid(
        basic_header("other-user", "correct-horse"),
        "demo-user",
        "correct-horse",
    )


def test_rejects_missing_or_malformed_headers():
    assert not is_basic_auth_valid(None, "demo-user", "correct-horse")
    assert not is_basic_auth_valid("Bearer abc", "demo-user", "correct-horse")
    assert not is_basic_auth_valid("Basic not-base64", "demo-user", "correct-horse")
    assert not is_basic_auth_valid("Basic ", "demo-user", "correct-horse")


def test_rejects_incomplete_configuration_and_colon_username():
    header = basic_header("demo-user", "correct-horse")
    assert not is_basic_auth_valid(header, "", "correct-horse")
    assert not is_basic_auth_valid(header, "demo-user", "")
    assert not is_basic_auth_valid(header, "bad:user", "correct-horse")
