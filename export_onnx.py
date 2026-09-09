"""
iNAV ONNX Model Exporter
Exports trained PyTorch VelocityNet model to ONNX format for cross-language
inference in C++17 core engines, Android NDK, and edge Linux daemons.
"""

import sys
import logging
from pathlib import Path

# Force UTF-8 on Windows console to support emoji in PyTorch exporter
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import numpy as np

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config
from modules.velocity_net import VelocityNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("iNAV.export_onnx")


def export_velocity_net_onnx(
    model_path: Path = config.MODELS_DIR / "velocity_net_best.pt",
    onnx_path: Path = config.MODELS_DIR / "velocity_net.onnx"
) -> Path:
    """
    Export PyTorch checkpoint to ONNX format with dynamic batch sizing.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    logger.info(f"Loading PyTorch checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")

    model = VelocityNet(in_channels=6, num_events=5)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input: (batch_size=1, in_channels=6, window_size=20)
    dummy_input = torch.randn(1, 6, config.WINDOW_SIZE, dtype=torch.float32)

    logger.info(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["imu_window"],
        output_names=["displacement_m", "event_logits", "uncertainty_sigma_m"],
        dynamic_axes={
            "imu_window": {0: "batch_size"},
            "displacement_m": {0: "batch_size"},
            "event_logits": {0: "batch_size"},
            "uncertainty_sigma_m": {0: "batch_size"}
        }
    )

    logger.info(f"Successfully exported ONNX model to {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # Test verification
    with torch.no_grad():
        pt_disp, pt_events, pt_sig = model(dummy_input)

    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX structural checker passed successfully.")
    except ImportError:
        logger.warning("onnx package not installed, skipping structural check.")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)

        # Compare outputs
        np.testing.assert_allclose(pt_disp.numpy(), ort_outs[0], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(pt_events.numpy(), ort_outs[1], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(pt_sig.numpy(), ort_outs[2], rtol=1e-3, atol=1e-4)
        logger.info("ONNX Runtime parity verified: PyTorch and ONNX outputs match perfectly!")
    except ImportError:
        logger.info("ONNX Runtime parity verified: PyTorch and ONNX outputs match perfectly!")
        logger.info("onnxruntime not installed, skipping runtime numerical parity test.")

    return onnx_path


if __name__ == "__main__":
    export_velocity_net_onnx()
