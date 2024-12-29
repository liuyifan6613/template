import numpy as np
from PIL import Image

from .loop import InferenceLoop

from ..pipeline import (
    SRNetPipeline,
)

class SRInferenceLoop(InferenceLoop):

    def load_pipeline(self) -> None:
        self.pipeline = SRNetPipeline(
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
            self.args.upscale,
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
