import pathlib
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class GoldenVerifyTest(unittest.TestCase):
    def test_c_output_matches_python_golden(self) -> None:
        build = subprocess.run(
            ["make", "-C", str(REPO_ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            build.returncode,
            msg=f"build failed\nstdout:\n{build.stdout}\nstderr:\n{build.stderr}",
        )

        verify = subprocess.run(
            [
                "python3",
                str(REPO_ROOT / "tools" / "golden_verify.py"),
                str(REPO_ROOT / "random_token_demo"),
                "64",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
