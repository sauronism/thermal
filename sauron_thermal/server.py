import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from sauron_thermal.camera import Camera
from sauron_thermal.thermal_camera import thermal_camera

app = FastAPI()

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
cv_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cv")


def to_jpeg(frame: np.ndarray) -> bytes:
    """Convert frame to jpeg."""
    _, raw = cv2.imencode(ext=".jpg", img=frame)
    return raw.tobytes()


async def frame_generator(camera: Camera):
    """Frame generator.

    Encodes the frames as multipart/x-mixed-replace.

    """
    async for raw in camera:
        jpeg = await asyncio.get_running_loop().run_in_executor(
            cv_executor,
            to_jpeg,
            raw,
        )
        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )


@app.get("/feed/{device_id}/")
async def feed(device_id: int = 0):
    camera = thermal_camera(device=device_id)
    return StreamingResponse(
        frame_generator(camera),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
