import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import (
    custom_all_reduce_v2 as custom_ar_module,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _UnreadableGraphMode:
    @property
    def _graph_mode_allowed(self):
        raise AssertionError(
            "_graph_mode_allowed must not be guarded in tc_piecewise mode"
        )


class TestCustomAllReduceV2ControlFlow(CustomTestCase):
    def test_tc_piecewise_short_circuits_mutable_capture_state(self):
        with (
            patch.object(
                custom_ar_module,
                "is_in_tc_piecewise_cuda_graph",
                return_value=True,
            ),
            patch.object(
                custom_ar_module.torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError(
                    "CUDA capture state must not be queried in tc_piecewise mode"
                ),
            ) as mock_is_capturing,
        ):
            self.assertFalse(CustomAllReduceV2._can_use_graph(_UnreadableGraphMode()))

        mock_is_capturing.assert_not_called()

    def test_non_piecewise_capture_still_uses_graph_mode(self):
        communicator = SimpleNamespace(_graph_mode_allowed=True)
        with (
            patch.object(
                custom_ar_module,
                "is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
            patch.object(
                custom_ar_module.torch.cuda,
                "is_current_stream_capturing",
                return_value=True,
            ) as mock_is_capturing,
        ):
            self.assertTrue(CustomAllReduceV2._can_use_graph(communicator))

        mock_is_capturing.assert_called_once_with()

    def test_disabled_graph_mode_skips_cuda_capture_query(self):
        communicator = SimpleNamespace(_graph_mode_allowed=False)
        with (
            patch.object(
                custom_ar_module,
                "is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
            patch.object(
                custom_ar_module.torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError(
                    "CUDA capture state must not be queried outside capture()"
                ),
            ) as mock_is_capturing,
        ):
            self.assertFalse(CustomAllReduceV2._can_use_graph(communicator))

        mock_is_capturing.assert_not_called()


if __name__ == "__main__":
    unittest.main()
