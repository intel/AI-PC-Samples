from pathlib import Path

import onnx
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "google/mobilenet_v2_1.0_224"
MODEL_REVISION = "75e607b00aeae1297cc89d026a118bce012f5c5a"
OUTPUT_DIRECTORY = Path("google_mobilenet_v2_1.0_224")
ONNX_MODEL_PATH = OUTPUT_DIRECTORY / "model.onnx"


class LogitsModel(torch.nn.Module):
    """Expose the tensor consumed by ONNX Runtime instead of a model output object."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    model.eval()

    crop_size = image_processor.crop_size
    height = int(crop_size["height"])
    width = int(crop_size["width"])
    pixel_values = torch.zeros(
        (1, model.config.num_channels, height, width),
        dtype=torch.float32,
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    torch.onnx.export(
        LogitsModel(model),
        (pixel_values,),
        str(ONNX_MODEL_PATH),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(ONNX_MODEL_PATH))
    model.config.save_pretrained(OUTPUT_DIRECTORY)
    image_processor.save_pretrained(OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
