import unittest
from unittest.mock import patch

from sglang.srt.compilation import (
    torch_compile_decoration as torch_compile_decoration_module,
)
from sglang.srt.model_executor.runner_backend import (
    tc_piecewise_cuda_graph_backend as backend_module,
)
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    TcPiecewiseCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTcPiecewiseCudaGraphBackend(CustomTestCase):
    def test_install_compile_configures_torch_before_install(self):
        events = []
        language_model = object()
        compile_config = object()
        graph_pool = object()

        with (
            patch.object(
                torch_compile_decoration_module,
                "set_torch_compile_config",
                side_effect=lambda: events.append("config"),
            ) as mock_set_config,
            patch.object(
                backend_module,
                "install_torch_compiled",
                side_effect=lambda *args, **kwargs: events.append("install"),
            ) as mock_install,
        ):
            TcPiecewiseCudaGraphBackend.install_compile(
                language_model,
                compile_config=compile_config,
                graph_pool=graph_pool,
            )

        self.assertEqual(events, ["config", "install"])
        mock_set_config.assert_called_once_with()
        mock_install.assert_called_once_with(
            language_model,
            fullgraph=True,
            dynamic_arg_dims=None,
            compile_config=compile_config,
            graph_pool=graph_pool,
        )


if __name__ == "__main__":
    unittest.main()
