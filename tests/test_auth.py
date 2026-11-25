"""Unit tests for the HTTP Basic authentication helper."""
import base64
import unittest

from auth import basic_auth_valid


def basic_header(username, password):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


class BasicAuthTests(unittest.TestCase):
    def test_accepts_valid_credentials(self):
        self.assertTrue(basic_auth_valid(basic_header("demo", "secret"), "demo", "secret"))

    def test_rejects_wrong_password(self):
        self.assertFalse(basic_auth_valid(basic_header("demo", "wrong"), "demo", "secret"))

    def test_rejects_wrong_username(self):
        self.assertFalse(basic_auth_valid(basic_header("other", "secret"), "demo", "secret"))

    def test_rejects_missing_or_malformed_header(self):
        for header in ("", "Bearer abc", "Basic !!!", "Basic "):
            self.assertFalse(basic_auth_valid(header, "demo", "secret"))

    def test_rejects_missing_server_credentials(self):
        self.assertFalse(basic_auth_valid(basic_header("demo", "secret"), "", "secret"))
        self.assertFalse(basic_auth_valid(basic_header("demo", "secret"), "demo", ""))

    def test_password_may_contain_colons(self):
        self.assertTrue(basic_auth_valid(basic_header("demo", "a:b:c"), "demo", "a:b:c"))


if __name__ == "__main__":
    unittest.main()
