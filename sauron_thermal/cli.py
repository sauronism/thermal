import asyncio
import functools


import typer
from typer import Typer
from datetime import datetime

from sauron_thermal.display import DisplayConfig, DisplayQuit

app = Typer()


def typer_async(_app: Typer, *_args, **_kwargs):
    """Decorator to allow async functions to be used as Typer commands."""

    def decorator(func):
        """Decorator."""

        @_app.command(*_args, **_kwargs)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper."""
            return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))

        return wrapper

    return decorator


@typer_async(app)
async def main(
        display: bool = True,
        save: bool = False,
        device: int = 0,
):
    """Sauron Thermal"""
    from sauron_thermal.thermal_camera import thermal_camera
    from sauron_thermal.display import Display

    if save:
        raise NotImplementedError("Saving to file not implemented yet.")

    camera = thermal_camera(device=device)
    if not display:
        raise NotImplementedError("Running without display not implemented yet.")

    display = Display(config=DisplayConfig(
        width=camera.config.width,
        height=camera.config.height,
        title="Sauron Thermal",
    ))

    try:
        with display:
            async for frame in camera:
                typer.echo(f"Frame: {datetime.now()}")
                await display.arender(frame)
            else:
                typer.echo("Camera stream ended.")
    except KeyboardInterrupt:
        typer.echo("Keyboard interrupt. Quitting...")
        pass
    except DisplayQuit:
        typer.echo("Detected display quit. Quitting...")
        pass


if __name__ == "__main__":
    asyncio.run(app())
