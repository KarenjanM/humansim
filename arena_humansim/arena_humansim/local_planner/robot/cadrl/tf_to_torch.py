# Adapted from mit-acl/cadrl_ros / mit-acl/gym-collision-avoidance
# (MIT license, Copyright (c) 2018 MIT-ACL).
#
# Walks an IROS18 GA3C-CADRL TF1 checkpoint and converts the variable tensors
# into a PyTorch state_dict that loads cleanly into model.GA3CCADRLNet.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .model import GA3CCADRLNet

# TF dense weight is (in, out); torch Linear is (out, in) -> transpose.
# TF LSTM kernel order is (i, j, f, o); torch LSTM is (i, f, g, o) -> reorder.
_TF_LSTM_KERNEL = "rnn/lstm_cell/kernel"
_TF_LSTM_BIAS = "rnn/lstm_cell/bias"
_TF_DENSE_LAYERS = {
    "layer1": "layer1",
    "layer2": "layer2",
    "fullyconnected1": "fullyconnected1",
    "logits_p": "logits_p",
    "logits_v": "logits_v",
}


def _list_tf_vars(prefix: str) -> dict[str, np.ndarray]:
    import tensorflow as tf

    reader = tf.train.load_checkpoint(prefix)
    shapes = reader.get_variable_to_shape_map()
    out: dict[str, np.ndarray] = {}
    for raw_name in shapes:
        # TF1 checkpoint reader returns names with a `:0` tensor-output suffix.
        canonical = raw_name.split(":", 1)[0]
        if "Adam" in canonical or canonical.endswith("_power") or canonical in ("global_step", "step"):
            continue
        out[canonical] = reader.get_tensor(raw_name)
    return out


def _convert_lstm(kernel: np.ndarray, bias: np.ndarray, input_size: int) -> dict[str, torch.Tensor]:
    # kernel shape: (input_size + hidden_size, 4 * hidden_size). Split rows into x and h chunks.
    kernel_x = kernel[:input_size, :]
    kernel_h = kernel[input_size:, :]

    def _reorder(w: np.ndarray) -> np.ndarray:
        i, j, f, o = np.split(w, 4, axis=1)
        return np.concatenate([i, f, j, o], axis=1)

    kernel_x = _reorder(kernel_x)
    kernel_h = _reorder(kernel_h)
    bias_re = _reorder(bias.reshape(1, -1)).reshape(-1)

    weight_ih = torch.from_numpy(kernel_x.T.copy()).float()
    weight_hh = torch.from_numpy(kernel_h.T.copy()).float()
    bias_ih = torch.from_numpy(bias_re.copy()).float()
    bias_hh = torch.zeros_like(bias_ih)
    return {
        "lstm.weight_ih_l0": weight_ih,
        "lstm.weight_hh_l0": weight_hh,
        "lstm.bias_ih_l0": bias_ih,
        "lstm.bias_hh_l0": bias_hh,
    }


def _convert_dense(tf_kernel: np.ndarray, tf_bias: np.ndarray, target: str) -> dict[str, torch.Tensor]:
    weight = torch.from_numpy(tf_kernel.T.copy()).float()
    bias = torch.from_numpy(tf_bias.copy()).float()
    return {f"{target}.weight": weight, f"{target}.bias": bias}


def convert_and_save(tf_prefix: str, pt_path: Path, num_actions: int = 11, max_other_agents: int = 10) -> None:
    tf_vars = _list_tf_vars(tf_prefix)

    state: dict[str, torch.Tensor] = {}

    if _TF_LSTM_KERNEL in tf_vars and _TF_LSTM_BIAS in tf_vars:
        kernel = tf_vars[_TF_LSTM_KERNEL]
        bias = tf_vars[_TF_LSTM_BIAS]
        hidden = bias.shape[0] // 4
        input_size = kernel.shape[0] - hidden
        state.update(_convert_lstm(kernel, bias, input_size))
    else:
        raise RuntimeError(f"LSTM tensors not found in checkpoint; got names: {sorted(tf_vars.keys())}")

    for tf_name, torch_name in _TF_DENSE_LAYERS.items():
        k = f"{tf_name}/kernel"
        b = f"{tf_name}/bias"
        if k not in tf_vars or b not in tf_vars:
            raise RuntimeError(f"Missing TF dense vars for {tf_name}: {k}, {b}")
        state.update(_convert_dense(tf_vars[k], tf_vars[b], torch_name))

    model = GA3CCADRLNet(num_actions=num_actions, max_other_agents=max_other_agents)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys after conversion: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing keys after conversion: {missing}")

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "num_actions": num_actions, "max_other_agents": max_other_agents}, pt_path)


def _self_test() -> None:
    model = GA3CCADRLNet()
    model.eval()
    host = torch.zeros(1, 4)
    other = torch.zeros(1, 10, 7)
    seq_len = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        logits_p, logits_v = model(host, other, seq_len)
    print("untrained logits_p[0]:", logits_p[0].numpy())
    print("untrained logits_v:", logits_v.item())


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        prefix = sys.argv[1]
        out = Path(sys.argv[2])
        convert_and_save(prefix, out)
        print(f"Wrote {out}")
    else:
        _self_test()
        print("CADRL tf_to_torch self-test OK")
