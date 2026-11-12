"""Run a vision model on whatever this computer has: .onnx on the CPU, .hef on the Hailo AI HAT+.

Both kinds take the same thing - an RGB uint8 image already resized to the model's input - so the
rest of the code does not care which one is loaded.
"""

import ast
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("yogurt.infer")


class OnnxModel:
    hailo = False

    def __init__(self, path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.height, self.width = self.session.get_inputs()[0].shape
        self.meta = dict(self.session.get_modelmeta().custom_metadata_map)

    def run(self, rgb: np.ndarray) -> list[np.ndarray]:
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return self.session.run(None, {self.input_name: x})


class HailoModel:
    """Follows Hailo's reference HailoInfer class (Hailo-Application-Code-Examples, runtime/python/common)."""

    hailo = True

    def __init__(self, path: Path):
        from hailo_platform import HEF, FormatType, HailoSchedulingAlgorithm, VDevice

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN  # lets several models share the chip
        params.group_id = "SHARED"
        self.device = VDevice(params)
        self.model = self.device.create_infer_model(str(path))
        self.model.set_batch_size(1)
        self.outputs = [info.name for info in HEF(str(path)).get_output_vstream_infos()]
        for name in self.outputs:
            self.model.output(name).set_format_type(FormatType.FLOAT32)
        self._context = self.model.configure()
        self.configured = self._context.__enter__()
        self.height, self.width = self.model.input().shape[:2]
        self.meta = {}

    def run(self, rgb: np.ndarray) -> list:
        buffers = {name: np.empty(self.model.output(name).shape, dtype=np.float32) for name in self.outputs}
        bindings = self.configured.create_bindings(output_buffers=buffers)
        bindings.input().set_buffer(np.ascontiguousarray(rgb, dtype=np.uint8))
        self.configured.wait_for_async_ready(timeout_ms=10000)
        self.configured.run_async([bindings], lambda completion_info: None).wait(10000)
        return [bindings.output(name).get_buffer() for name in self.outputs]


def load(path: Path):
    """Load the model at path, or return None if it has not been trained/exported yet."""
    path = Path(path)
    if not path.exists():
        return None
    model = HailoModel(path) if path.suffix == ".hef" else OnnxModel(path)
    sidecar = path.with_suffix(".json")  # labels, thresholds... written next to the model by the trainers
    if sidecar.exists():
        model.meta.update(json.loads(sidecar.read_text()))
    if isinstance(model.meta.get("names"), str):  # Ultralytics stores its label map as text
        model.meta["names"] = ast.literal_eval(model.meta["names"])
    log.info("loaded %s (%s, input %dx%d)", path.name, "Hailo" if model.hailo else "CPU", model.width, model.height)
    return model
