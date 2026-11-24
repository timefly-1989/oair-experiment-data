"""Run checks on archived OAIR operators using synthetic tensors on CPU."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_operator_source():
    path = ROOT / "experiments/cap20/garment_study/model.py"
    spec = spec_from_file_location("oair_archived_operators", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    model = load_operator_source()
    torch.manual_seed(2110)
    width = 8
    anchor, state, evidence = (torch.randn(5, width) for _ in range(3))
    boundary = torch.ones(5)
    visibility = torch.ones(5)
    layer = model.Delta(width, anchored=True).eval()
    write_values = []
    handle = layer.value.register_forward_hook(
        lambda _, args, output: write_values.append(output.detach().clone()))
    with torch.inference_mode():
        before = layer(state, evidence, anchor, boundary, visibility)
        after = layer(state + 0.2 * torch.randn_like(state), evidence * 2,
                      anchor, boundary, visibility)
    handle.remove()
    torch.testing.assert_close(write_values[0], write_values[1], rtol=0, atol=0)
    assert not torch.equal(before, after), "The state still affects the updated features"

    vertices = torch.tensor([[0., 0., 2.], [0., 0., 2.], [0., 0., 2.]])
    displacement = torch.tensor([[0.04, -0.06, 0.08]]).repeat(3, 1)
    protected = model.protect_boundary_projection(
        displacement, vertices, torch.tensor([1., 0., 1.]), torch.tensor([1., 1., 0.]))
    expected = torch.tensor([[0., 0., 0.08], [0.04, -0.06, 0.08], [0.04, -0.06, 0.08]])
    torch.testing.assert_close(protected, expected)
    print(f"PyTorch {torch.__version__}: fixed-anchor write and camera-ray projection checks passed.")


if __name__ == "__main__":
    main()
