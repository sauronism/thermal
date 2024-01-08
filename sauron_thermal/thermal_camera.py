"""
The thermal camera module.

We apply the following processing steps to the raw image:
- Convert from 16-bit ints to 32-bit float.
- Scale to the range [0, 1].
- Apply AGC (Automatic Gain Control).
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

"""

from sauron_thermal.agc import AGC, AgcConfig
from sauron_thermal.camera import Camera, CameraConfig
from sauron_thermal.clahe import ClaheConfig, CLAHE
from sauron_thermal.simple_agc import SimpleAgc, SimpleAgcConfig
from sauron_thermal.to_float import ToFloat32


def thermal_camera(
        device: int,
        agc_config: AgcConfig = AgcConfig(
            sample_rate=9.0,
            attack_time=0.1,
            release_time=0.1,
            reference=0.1,
            gain_floor=0.1,
            gain_ceiling=10.0,
        ),
        clahe_config: ClaheConfig = ClaheConfig(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
        ),

) -> Camera:
    """Thermal camera module."""
    boson_config = CameraConfig(
        device=device,
        width=640,
        height=512,
        fps=9,

    )

    return Camera(
        config=boson_config,
        filters=[
            ToFloat32(),
            SimpleAgc(config=SimpleAgcConfig()),
            # AGC(config=agc_config),
            # CLAHE(config=clahe_config),
        ],
    )
