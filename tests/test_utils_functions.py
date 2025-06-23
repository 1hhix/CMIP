import os
import json
import tempfile
import torch
from utils.functions import load_args, do_batch_rep

def test_load_args():
    # Create a temporary JSON file
    args = {"problem": "mtsp_foo"}
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        json.dump(args, f)
        fname = f.name
    loaded = load_args(fname)
    assert loaded["problem"] == "mtsp"
    assert "data_distribution" in loaded
    os.remove(fname)

def test_do_batch_rep_tensor():
    t = torch.randn(2, 3)
    out = do_batch_rep(t, 4)
    assert out.shape[0] == 8

def test_do_batch_rep_dict():
    d = {"a": torch.randn(2, 3)}
    out = do_batch_rep(d, 2)
    assert out["a"].shape[0] == 4 