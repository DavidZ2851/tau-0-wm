#!/usr/bin/env python3
"""
Closed-loop inference script for the Tau-0 world model policy.

Connects to the tau inference server (web_infer_utils/server.py) and runs a
paced control loop with open-loop chunk execution and temporal aggregation,
mirroring the structure of hermes/run_robot_inference.py.

Usage:
    python scripts/infer.py --host 127.0.0.1 --port 8001 --prompt "pick up the cup"
"""

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from web_infer_utils.openpi_client.websocket_client_policy import WebsocketClientPolicy


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop tau policy inference client")

    # Server
    parser.add_argument("--host", default="127.0.0.1", help="Policy server host")
    parser.add_argument("--port", default=8001, type=int, help="Policy server port")

    # Task
    parser.add_argument("--prompt", default="Action", help="Task description")
    parser.add_argument("--num-inference-steps", type=int, default=5,
                        help="Denoising steps per inference call (default: 5)")
    parser.add_argument("--execution-step", type=int, default=30,
                        help="Max action steps returned per call (default: 30)")
    parser.add_argument("--shift", type=float, default=1.0, help="Flow matching shift")
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "euler"])

    # Control loop
    parser.add_argument("--open-loop-horizon", type=int, default=6,
                        help="Steps to execute from each chunk before re-inferring (default: 6)")
    parser.add_argument("--control-fps", type=float, default=15.0,
                        help="Target control rate in Hz (default: 15)")
    parser.add_argument("--no-temporal-agg", action="store_true",
                        help="Disable temporal aggregation; execute raw chunk actions")
    parser.add_argument("--exp-weight", type=float, default=0.01,
                        help="Exponential weight for temporal aggregation (default: 0.01)")

    # Saving
    parser.add_argument("--save-folder", default=None,
                        help="Directory to save actions and images")
    parser.add_argument("--save-queue-size", type=int, default=100,
                        help="Async save queue depth (default: 100)")

    # Misc
    parser.add_argument("--action-dim", type=int, default=20,
                        help="Action dimension (must match server config, default: 20)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many control steps (default: run until Ctrl+C)")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ctrl-C handler (mirrors hermes)
# ---------------------------------------------------------------------------

@contextmanager
def ctrl_c_handler(signal_handler=None):
    class _State:
        def __init__(self):
            self._caught = False
        def __bool__(self):
            return self._caught

    state = _State()

    def _handler(sig, frame):
        state._caught = True
        if signal_handler:
            signal_handler()

    original = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)
    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, original)


# ---------------------------------------------------------------------------
# Async data writer (mirrors hermes AsyncDataWriter)
# ---------------------------------------------------------------------------

class AsyncDataWriter:
    """Background-thread writer that never blocks the control loop."""

    def __init__(self, save_folder: str, queue_size: int = 100):
        self.save_folder = save_folder
        self._q = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._running = False
        self._dropped = 0
        self._written = 0
        self._actions_f = None

    def start(self):
        os.makedirs(self.save_folder, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            self._actions_f = open(
                os.path.join(self.save_folder, "actions.jsonl"), "w", buffering=1
            )
            while self._running or not self._q.empty():
                try:
                    item = self._q.get(timeout=0.1)
                    if item is None:
                        break
                    ts, actions, obs_image = item
                    self._actions_f.write(
                        json.dumps({"ts": ts, "actions": actions.tolist()}) + "\n"
                    )
                    if obs_image is not None:
                        frame_path = os.path.join(self.save_folder, f"{ts:.6f}.jpg")
                        cv2.imwrite(frame_path, cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR))
                    self._written += 1
                    self._q.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[AsyncWriter] Error: {e}")
        finally:
            if self._actions_f:
                self._actions_f.close()

    def write_async(self, ts: float, actions: np.ndarray, obs_image: np.ndarray | None):
        try:
            self._q.put_nowait((ts, actions, obs_image))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 10 == 0:
                print(f"[AsyncWriter] Queue full — dropped {self._dropped} frames so far")

    def stop(self):
        self._running = False
        print(f"Flushing writer queue ({self._q.qsize()} items)...")
        try:
            self._q.join()
        except Exception:
            pass
        try:
            self._q.put(None, timeout=1.0)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)
        print(f"[AsyncWriter] {self._written} written, {self._dropped} dropped")
        if self._dropped > 0:
            print(f"  Consider increasing --save-queue-size (current: {self._q.maxsize})")


# ---------------------------------------------------------------------------
# Policy inference runner (mirrors hermes PolicyInference)
# ---------------------------------------------------------------------------

class PolicyInference:
    """
    Closed-loop policy runner with open-loop chunk execution and optional
    temporal aggregation (identical design to hermes PolicyInference).
    """

    def __init__(
        self,
        prompt: str,
        open_loop_horizon: int = 6,
        control_fps: float = 15.0,
        temporal_agg: bool = True,
        exp_weight: float = 0.01,
        action_dim: int = 20,
        save_folder: str = None,
        save_queue_size: int = 100,
        verbose: bool = False,
    ):
        self.prompt = prompt
        self.open_loop_horizon = max(1, open_loop_horizon)
        self.control_fps = control_fps
        self.control_period = 1.0 / control_fps if control_fps > 0 else 0.0
        self.temporal_agg = temporal_agg
        self.exp_weight = exp_weight
        self.action_dim = action_dim
        self.verbose = verbose

        self.running = True

        # Open-loop chunk state
        self.robot_step = 0
        self.query_idx = 0
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None

        # Temporal aggregation buffer
        self._max_timesteps = 1000
        self._chunk_size: int | None = None
        self._all_time_actions: np.ndarray | None = None
        self._exp_weights_cache: dict[int, np.ndarray] = {}
        self._temporal_lock = threading.Lock()

        # Latency tracking
        self.latency: dict[str, list] = defaultdict(list)

        # Async writer
        self.async_writer = None
        if save_folder:
            self.async_writer = AsyncDataWriter(save_folder, save_queue_size)
            self.async_writer.start()

    # ------------------------------------------------------------------
    # Temporal aggregation
    # ------------------------------------------------------------------

    def _aggregate_actions(self, step: int) -> np.ndarray | None:
        with self._temporal_lock:
            if self._all_time_actions is None or step >= self._max_timesteps:
                return None
            actions_for_step = self._all_time_actions[:, step, :]
            populated = np.any(actions_for_step != 0, axis=1)
            actions_for_step = actions_for_step[populated]
            if len(actions_for_step) == 0:
                return None
            n = len(actions_for_step)
            if n not in self._exp_weights_cache:
                w = np.exp(-self.exp_weight * np.arange(n - 1, -1, -1))
                self._exp_weights_cache[n] = w / w.sum()
            return (actions_for_step * self._exp_weights_cache[n][:, None]).sum(axis=0)

    def _update_temporal_buffer(self, chunk: np.ndarray, query_idx: int, start: int):
        with self._temporal_lock:
            if self._all_time_actions is None:
                self._chunk_size = len(chunk)
                self._all_time_actions = np.zeros(
                    (self._max_timesteps, self._max_timesteps, self.action_dim), dtype=np.float32
                )
            end = min(start + self._chunk_size, self._max_timesteps)
            length = end - start
            if length > 0 and query_idx < self._max_timesteps:
                self._all_time_actions[query_idx, start:end, :] = chunk[:length]

    # ------------------------------------------------------------------
    # Replanning
    # ------------------------------------------------------------------

    def _needs_replan(self) -> bool:
        if self.pred_action_chunk is None:
            return True
        return self.actions_from_chunk_completed >= min(
            self.open_loop_horizon, len(self.pred_action_chunk)
        )

    def _run_inference(
        self,
        policy_client: WebsocketClientPolicy,
        obs: np.ndarray,
        state: np.ndarray,
        gripper_states: np.ndarray,
        num_inference_steps: int,
        execution_step: int,
        shift: float,
        sample_solver: str,
    ):
        t0 = time.perf_counter()

        payload = {
            "obs": obs,
            "prompt": self.prompt,
            "state": state,
            "gripper_states": gripper_states,
            "num_inference_steps": num_inference_steps,
            "execution_step": execution_step,
            "shift": shift,
            "sample_solver": sample_solver,
        }

        t_infer = time.perf_counter()
        result = policy_client.infer(obs=payload)
        actions = result["actions"]  # (T, action_dim)
        t_infer_end = time.perf_counter()
        self.latency["policy"].append(t_infer_end - t_infer)

        if self.temporal_agg:
            self._update_temporal_buffer(actions, self.query_idx, self.robot_step)
            self.query_idx += 1

        self.pred_action_chunk = actions
        self.actions_from_chunk_completed = 0
        self.latency["inference(total)"].append(t_infer_end - t0)

        if self.async_writer is not None:
            # Save the ego image (first view) alongside the predicted chunk
            ego_img = None
            if obs.ndim == 4:  # (V, H, W, 3)
                ego_img = obs[0]
            self.async_writer.write_async(t0, actions, ego_img)

        if self.verbose:
            print(
                f"  replan step={self.robot_step} chunk={actions.shape} "
                f"policy={t_infer_end - t_infer:.3f}s"
            )

    # ------------------------------------------------------------------
    # Control step (called once per tick)
    # ------------------------------------------------------------------

    def run_control_step(
        self,
        policy_client: WebsocketClientPolicy,
        obs: np.ndarray,
        state: np.ndarray,
        gripper_states: np.ndarray,
        num_inference_steps: int = 5,
        execution_step: int = 30,
        shift: float = 1.0,
        sample_solver: str = "unipc",
    ) -> np.ndarray | None:
        if not self.running:
            return None

        if self._needs_replan():
            self._run_inference(
                policy_client, obs, state, gripper_states,
                num_inference_steps, execution_step, shift, sample_solver,
            )

        if self.pred_action_chunk is None:
            return None

        chunk_idx = self.actions_from_chunk_completed
        if self.temporal_agg:
            action = self._aggregate_actions(self.robot_step)
            if action is None:
                action = self.pred_action_chunk[min(chunk_idx, len(self.pred_action_chunk) - 1)]
        else:
            action = self.pred_action_chunk[min(chunk_idx, len(self.pred_action_chunk) - 1)]

        self.actions_from_chunk_completed += 1
        self.robot_step += 1
        return action

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self):
        with self._temporal_lock:
            if self._all_time_actions is not None:
                self._all_time_actions.fill(0)
        self.robot_step = 0
        self.query_idx = 0
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        if self.verbose:
            print("Episode state reset")

    def shutdown(self):
        self.running = False
        if self.async_writer is not None:
            self.async_writer.stop()

    def __del__(self):
        if not self.latency:
            return
        ncols = 24
        rows = []
        for key, vals in self.latency.items():
            mean = np.mean(vals)
            hz = 1.0 / mean if mean > 0 else 0.0
            rows.append(f"| {key.ljust(ncols)} | {mean:.5f}s ({hz:8.2f} Hz) |")
        if rows:
            w = len(rows[0])
            print("\n" + "-" * w)
            print(f"| {'Tau policy inference latency'.center(w - 4)} |")
            print("-" * w)
            for r in rows:
                print(r)
            print("-" * w + "\n")


# ---------------------------------------------------------------------------
# Observation / robot state hooks — replace with real hardware
# ---------------------------------------------------------------------------

def get_observation(img_size=(192, 256)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (obs, state, gripper_states) for one control tick.

    Replace this stub with your actual camera and robot-state readers.
    obs          — (V, H, W, 3) uint8, views in order: top_head, hand_left, hand_right
    state        — (14,) float32, arm EEF pose (xyz + quaternion) × 2 arms
    gripper_states — (2,) float32 in [0, 120]
    """
    H, W = img_size
    obs = (np.random.rand(3, H, W, 3) * 255).astype(np.uint8)
    state = np.random.rand(14).astype(np.float32)
    gripper_states = (np.random.rand(2) * 120).astype(np.float32)
    return obs, state, gripper_states


def apply_action(action: np.ndarray):
    """Send the 20D action to your robot controller. Replace this stub."""
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Connecting to policy server at {args.host}:{args.port} ...")
    policy_client = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Connected. Server metadata: {policy_client.get_server_metadata()}")

    runner = PolicyInference(
        prompt=args.prompt,
        open_loop_horizon=args.open_loop_horizon,
        control_fps=args.control_fps,
        temporal_agg=not args.no_temporal_agg,
        exp_weight=args.exp_weight,
        action_dim=args.action_dim,
        save_folder=args.save_folder,
        save_queue_size=args.save_queue_size,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("Tau Policy Inference")
    print("=" * 60)
    print(f"  Prompt:           '{args.prompt}'")
    print(f"  Control FPS:       {args.control_fps} ({runner.control_period * 1000:.1f} ms/step)")
    print(f"  Open-loop horizon: {args.open_loop_horizon} steps")
    print(f"  Temporal agg:      {not args.no_temporal_agg}")
    print(f"  Inference steps:   {args.num_inference_steps}")
    print(f"  Save folder:       {args.save_folder or 'disabled'}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    step = 0
    try:
        with ctrl_c_handler() as ctrl_c:
            next_tick = time.perf_counter()
            while not ctrl_c:
                if args.max_steps is not None and step >= args.max_steps:
                    break

                obs, state, gripper_states = get_observation()

                action = runner.run_control_step(
                    policy_client,
                    obs=obs,
                    state=state,
                    gripper_states=gripper_states,
                    num_inference_steps=args.num_inference_steps,
                    execution_step=args.execution_step,
                    shift=args.shift,
                    sample_solver=args.sample_solver,
                )

                if action is not None:
                    apply_action(action)
                    step += 1

                # Pace to control_fps
                if runner.control_period > 0:
                    next_tick += runner.control_period
                    wait = next_tick - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    elif wait < -runner.control_period:
                        next_tick = time.perf_counter()

    finally:
        print("\nShutting down...")
        runner.shutdown()
        print(f"Completed {step} control steps.")


if __name__ == "__main__":
    main()
