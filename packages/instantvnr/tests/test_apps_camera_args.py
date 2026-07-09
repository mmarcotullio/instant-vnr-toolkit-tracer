"""Unit tests for camera argument encoding in instantvnr.apps launchers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import instantvnr.apps as apps


@pytest.fixture
def capture_command(monkeypatch):
    """Capture the command a launcher would execute without running it."""
    captured = {}

    def fake_call(command, env=None):
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(apps, "_find_exe", lambda name: name)
    monkeypatch.setattr(apps, "_runtime_env", lambda: {})
    monkeypatch.setattr(apps.subprocess, "call", fake_call)
    return captured


def _arg_after(command, flag):
    index = command.index(flag)
    return command[index + 1]


def _args_after(command, flag, count):
    index = command.index(flag)
    return command[index + 1 : index + 1 + count]


@pytest.mark.parametrize(
    "camera_value",
    [
        [0, 1, -2],
        "0,1,-2",
        np.array([0, 1, -2], dtype=np.float32),
        torch.tensor([0, 1, -2], dtype=torch.float32),
    ],
)
def test_run_batch_accepts_common_vec3_inputs(capture_command, camera_value):
    apps.run_batch(
        neural_volume="scene.bson",
        camera_from=camera_value,
        camera_at=camera_value,
        camera_up=camera_value,
    )

    command = capture_command["command"]
    expected = "0.0,1.0,-2.0"
    assert _arg_after(command, "--camera-from") == expected
    assert _arg_after(command, "--camera-at") == expected
    assert _arg_after(command, "--camera-up") == expected


@pytest.mark.parametrize(
    "camera_value",
    [
        [0, 1, -2],
        "0,1,-2.0",
        np.array([0, 1, -2], dtype=np.float32),
        torch.tensor([0, 1, -2], dtype=torch.float32),
    ],
)
def test_run_int_single_accepts_common_vec3_inputs(capture_command, camera_value):
    apps.run_int_single(
        neural_volume="scene.bson",
        camera_from=camera_value,
        camera_at=camera_value,
        camera_up=camera_value,
    )

    command = capture_command["command"]
    expected = "0.0,1.0,-2.0"
    assert _arg_after(command, "--camera-from") == expected
    assert _arg_after(command, "--camera-at") == expected
    assert _arg_after(command, "--camera-up") == expected


@pytest.mark.parametrize(
    "camera_value",
    [
        [0, 1, -2],
        "0,1,-2",
        np.array([0, 1, -2], dtype=np.float32),
        torch.tensor([0, 1, -2], dtype=torch.float32),
    ],
)
def test_run_int_dual_accepts_common_vec3_inputs(capture_command, camera_value):
    apps.run_int_dual(
        volume="scene.json",
        network="network.json",
        camera_from=camera_value,
        camera_at=camera_value,
        camera_up=camera_value,
    )

    command = capture_command["command"]
    expected = ["0.0", "1.0", "-2.0"]
    assert _args_after(command, "--camera-from", 3) == expected
    assert _args_after(command, "--camera-at", 3) == expected
    assert _args_after(command, "--camera-up", 3) == expected

