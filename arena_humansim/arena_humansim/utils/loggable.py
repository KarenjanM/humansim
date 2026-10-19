from functools import cached_property

import rclpy.impl.rcutils_logger
import rclpy.node


class Loggable:
    _node: rclpy.node.Node | None = None

    @classmethod
    def init_logging(cls, node: rclpy.node.Node) -> None:
        cls._node = node

    @cached_property
    def _logger(self) -> rclpy.impl.rcutils_logger.RcutilsLogger:
        assert self._node is not None
        return self._node.get_logger().get_child(type(self).__qualname__)
