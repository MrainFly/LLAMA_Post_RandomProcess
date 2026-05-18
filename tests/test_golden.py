import pathlib
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class GoldenVerifyTest(unittest.TestCase):
    def run_verify(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(REPO_ROOT / "tools" / "golden_verify.py"),
                str(REPO_ROOT / "random_token_demo"),
                "64",
                *extra_args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_c_output_matches_python_golden_default(self) -> None:
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

        verify = self.run_verify()
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )

    def test_boundary_k1(self) -> None:
        verify = self.run_verify("--top-k", "1")
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )

    def test_boundary_top_p_one(self) -> None:
        verify = self.run_verify("--top-p", "1.0")
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )

    def test_boundary_top_k_zero(self) -> None:
        verify = self.run_verify("--top-k", "0")
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )

    def test_boundary_min_p_zero(self) -> None:
        verify = self.run_verify("--min-p", "0.0")
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )

    def test_strong_penalty(self) -> None:
        verify = self.run_verify(
            "--repeat-penalty",
            "2.0",
            "--frequency-penalty",
            "0.8",
            "--presence-penalty",
            "0.5",
        )
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )

    def test_history_window_zero_means_max_history(self) -> None:
        verify = self.run_verify("--history-window", "0")
        self.assertEqual(
            0,
            verify.returncode,
            msg=f"verify failed\nstdout:\n{verify.stdout}\nstderr:\n{verify.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
