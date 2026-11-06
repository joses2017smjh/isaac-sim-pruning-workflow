"""Bring up the ROS 2 side of the live Isaac loop: controller, C++ safety gate, optional panel.

Waits for the simulator's latched ``/pruning/controller/config``, starts
``safety_gate_node`` with that run's target and jaw geometry, runs the
``LiveController`` in this process, and optionally starts the Qt/QML operator
panel. Every gate decision (JSON mirror) is appended to
``<output>/ros2_loop_decisions.jsonl`` with its receive time, and a summary is
written to ``<output>/ros2_loop_bringup.json`` when the simulator sends its end
marker. Runs inside the SIL container on the same node as Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from . import topics as T
from .live_controller import LATCHED, RELIABLE, LiveController


class DecisionLog(Node):
    def __init__(self, path):
        super().__init__("pruning_loop_decision_log")
        self.stream = open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self.count = 0
        self.create_subscription(String, T.TOPIC_SAFETY_DECISION_JSON, self._on_decision, RELIABLE)

    def _on_decision(self, message):
        record = json.loads(message.data)
        record["log_receive_wall_ns"] = time.time_ns()
        self.stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.stream.flush()
        self.count += 1

    def close(self):
        self.stream.close()


class ConfigWaiter(Node):
    def __init__(self):
        super().__init__("pruning_loop_config_waiter")
        self.config = None
        self.create_subscription(String, T.TOPIC_CONTROLLER_CONFIG, self._on_config, LATCHED)

    def _on_config(self, message):
        self.config = json.loads(message.data)


def _array(values):
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


def start_gate(executable, config, input_timeout_s, log_path):
    command = [
        executable,
        "--ros-args",
        "-p", f"selected_target_id:={config['target_id']}",
        "-p", f"mouth_offset_tool:={_array(config['mouth_offset_tool_m'])}",
        "-p", f"closing_axis_tool:={_array(config['closing_axis_tool'])}",
        "-p", f"input_timeout_s:={float(input_timeout_s)}",
    ]  # fmt: skip
    stream = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - handed to the child
    return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True), stream


def stop(process, timeout=10):
    if process is None or process.poll() is not None:
        return None if process is None else process.returncode
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", required=True, help="installed safety_gate_node executable")
    parser.add_argument("--output-dir", type=Path, help="default: the run directory the simulator publishes")
    parser.add_argument("--input-timeout-s", type=float, default=10.0)
    parser.add_argument("--config-timeout-s", type=float, default=1800.0)
    parser.add_argument("--run-timeout-s", type=float, default=3600.0)
    parser.add_argument("--panel", nargs=argparse.REMAINDER, help="panel executable and its arguments")
    args = parser.parse_args(argv)

    rclpy.init()
    waiter = ConfigWaiter()
    started = time.monotonic()
    while waiter.config is None and time.monotonic() - started < args.config_timeout_s:
        rclpy.spin_once(waiter, timeout_sec=0.2)
    config = waiter.config
    waiter.destroy_node()
    summary = {"schema_version": 1, "config_received": config is not None}
    if config is None:
        print(json.dumps(summary), flush=True)
        rclpy.try_shutdown()
        return 2
    if args.output_dir is None:
        args.output_dir = Path(config["output_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gate, gate_log = start_gate(args.gate, config, args.input_timeout_s, args.output_dir / "safety_gate_node.log")
    panel = panel_log = None
    if args.panel:
        panel_log = open(args.output_dir / "operator_panel.log", "w", encoding="utf-8")  # noqa: SIM115
        # "{output_dir}" in the panel's arguments is the run directory, known only from the config.
        command = [part.replace("{output_dir}", str(args.output_dir)) for part in args.panel]
        panel = subprocess.Popen(command, stdout=panel_log, stderr=subprocess.STDOUT, text=True)

    controller = LiveController()
    log = DecisionLog(args.output_dir / "ros2_loop_decisions.jsonl")
    executor = SingleThreadedExecutor()
    executor.add_node(controller)
    executor.add_node(log)
    loop_started = time.monotonic()
    try:
        while rclpy.ok() and not controller.finished and time.monotonic() - loop_started < args.run_timeout_s:
            executor.spin_once(timeout_sec=0.05)
        # Drain the last decisions after the end marker.
        drain = time.monotonic()
        while time.monotonic() - drain < 2.0:
            executor.spin_once(timeout_sec=0.05)
    finally:
        timings = np.asarray(controller.timings or [0.0])
        summary.update(
            target_id=config.get("target_id"),
            ended_by_marker=controller.finished,
            controller_cycles=controller.cycles,
            controller_errors=controller.errors,
            controller_cycle_s={
                "p50": float(np.percentile(timings, 50)),
                "p95": float(np.percentile(timings, 95)),
                "max": float(timings.max()),
            },
            decisions_logged=log.count,
            gate_returncode=stop(gate),
            panel_returncode=stop(panel),
            input_timeout_s=args.input_timeout_s,
        )
        log.close()
        gate_log.close()
        if panel_log:
            panel_log.close()
        executor.shutdown()
        controller.destroy_node()
        log.destroy_node()
        rclpy.try_shutdown()
        (args.output_dir / "ros2_loop_bringup.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return 0 if controller.finished and not controller.errors else 1


if __name__ == "__main__":
    sys.exit(main())
