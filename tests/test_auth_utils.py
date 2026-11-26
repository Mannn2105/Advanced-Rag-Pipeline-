"""Unit tests for shared HTTP Basic credential validation."""
import base64
import unittest

from auth_utils import is_basic_auth_valid


def basic_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


class BasicAuthValidationTests(unittest.TestCase):
    def test_accepts_valid_basic_credentials(self):
        self.assertTrue(
            is_basic_auth_valid(
                basic_header("demo-user", "correct-horse"),
                "demo-user",
                "correct-horse",
            )
        )

    def test_accepts_colons_inside_password(self):
        self.assertTrue(
            is_basic_auth_valid(
                basic_header("demo-user", "part-one:part-two"),
                "demo-user",
                "part-one:part-two",
            )
        )

    def test_rejects_wrong_password(self):
        self.assertFalse(
            is_basic_auth_valid(
                basic_header("demo-user", "wrong"),
                "demo-user",
                "correct-horse",
            )
        )

    def test_rejects_wrong_username(self):
        self.assertFalse(
            is_basic_auth_valid(
                basic_header("other-user", "correct-horse"),
                "demo-user",
                "correct-horse",
            )
        )

    def test_rejects_missing_or_malformed_headers(self):
        self.assertFalse(is_basic_auth_valid(None, "demo-user", "correct-horse"))
        self.assertFalse(is_basic_auth_valid("Bearer abc", "demo-user", "correct-horse"))
        self.assertFalse(is_basic_auth_valid("Basic not-base64", "demo-user", "correct-horse"))
        self.assertFalse(is_basic_auth_valid("Basic ", "demo-user", "correct-horse"))

    def test_rejects_incomplete_configuration_and_colon_username(self):
        header = basic_header("demo-user", "correct-horse")
        self.assertFalse(is_basic_auth_valid(header, "", "correct-horse"))
        self.assertFalse(is_basic_auth_valid(header, "demo-user", ""))
        self.assertFalse(is_basic_auth_valid(header, "bad:user", "correct-horse"))


if __name__ == "__main__":
    unittest.main()
