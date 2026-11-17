"""Progress is streamed before the subprocess exits; failures keep partial counts."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from inspect_rooflang import run


class InspectProgressTests(unittest.TestCase):
    def test_progress_is_visible_before_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acknowledged = root / "ack"

            class Output(io.StringIO):
                def write(self, text):
                    if "已完成 1/2" in text:
                        acknowledged.touch()
                    return super().write(text)

            output = Output()
            code = """import time,json,sys
from pathlib import Path
ack=Path(sys.argv[1])
print('SIMTHESIZER_PROGRESS '+json.dumps(dict(completed=1,total=2,time_s=0.25)),flush=True)
for _ in range(100):
 if ack.exists(): break
 time.sleep(0.02)
else: raise RuntimeError('progress was buffered until process exit')
print('SIMTHESIZER_PROGRESS '+json.dumps(dict(completed=2,total=2,time_s=0.5)),flush=True)
"""
            with contextlib.redirect_stdout(output):
                run([sys.executable, "-c", code, str(acknowledged)], root / "run.log")
            self.assertIn("已完成 2/2", output.getvalue())
            self.assertIn("100%", output.getvalue())
            self.assertIn('"completed": 1', (root / "run.log").read_text())

    def test_failed_process_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            code = 'print(\'SIMTHESIZER_PROGRESS {"completed":1,"total":2,"time_s":0.25}\',flush=True); raise SystemExit(2)'
            with contextlib.redirect_stdout(output), self.assertRaises(RuntimeError):
                run([sys.executable, "-c", code], Path(directory) / "run.log")
            self.assertIn("已完成 1/2", output.getvalue())
            self.assertNotIn("100%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
