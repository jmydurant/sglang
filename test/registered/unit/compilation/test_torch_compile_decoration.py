import unittest
from contextlib import ExitStack
from unittest.mock import patch

from sglang.srt.compilation import torch_compile_decoration as compile_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTorchCompileDecoration(CustomTestCase):
    def test_sets_all_available_dynamo_recompile_limits(self):
        import torch._dynamo.config
        import torch._inductor.config

        limit_names = (
            "recompile_limit",
            "cache_size_limit",
            "accumulated_recompile_limit",
            "accumulated_cache_size_limit",
        )
        available_limit_names = [
            name for name in limit_names if hasattr(torch._dynamo.config, name)
        ]
        self.assertTrue(available_limit_names)

        inductor_options = (
            (torch._inductor.config, "coordinate_descent_tuning"),
            (torch._inductor.config.triton, "unique_kernel_names"),
            (torch._inductor.config, "fx_graph_cache"),
        )

        with ExitStack() as stack:
            for name in available_limit_names:
                stack.enter_context(patch.object(torch._dynamo.config, name, 1))
            for config, name in inductor_options:
                stack.enter_context(patch.object(config, name, getattr(config, name)))
            mock_monkey_patch = stack.enter_context(
                patch.object(compile_module, "monkey_patch_torch_compile")
            )

            compile_module.set_torch_compile_config()

            for name in available_limit_names:
                self.assertEqual(getattr(torch._dynamo.config, name), 1024)
            mock_monkey_patch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
