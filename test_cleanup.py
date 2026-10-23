import importlib.util
import signal
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class CompressionCleanupTests(unittest.TestCase):
    def test_model_error_restores_signal_and_closes_progress(self):
        progress = Mock()
        dependencies = {
            "numpy": types.ModuleType("numpy"),
            "llama_cpp": types.SimpleNamespace(Llama=Mock()),
            "more_itertools": types.SimpleNamespace(consume=lambda items: list(items)),
            "tqdm": types.SimpleNamespace(tqdm=Mock(return_value=progress)),
        }
        with patch.dict(sys.modules, dependencies):
            spec = importlib.util.spec_from_file_location(
                "llama_zip_cleanup_test", Path(__file__).with_name("llama_zip.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        compressor = module.LlamaZip.__new__(module.LlamaZip)
        compressor.verbose = False
        compressor.model = Mock()
        compressor.model.tokenize.return_value = []
        compressor.model.token_eos.return_value = 0
        compressor.model.generate.side_effect = RuntimeError("synthetic model failure")
        original_handler = signal.getsignal(signal.SIGINT)
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
                compressor.compress(b"")
            self.assertIs(signal.getsignal(signal.SIGINT), original_handler)
            progress.close.assert_called_once()
        finally:
            signal.signal(signal.SIGINT, original_handler)


if __name__ == "__main__":
    unittest.main()
