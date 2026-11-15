"""Open an image file the way the demo sees an upload.

Cameras and phones often store a photo unrotated and record the rotation in an
EXIF tag. Browsers and the demo's upload widget apply that tag; ``Image.open``
alone does not. A reference or test photo embedded without it is embedded
sideways, so it no longer matches the same photo uploaded to the demo. Scripts
that embed image files should open them with ``open_upright``.
"""

from __future__ import annotations

import os

from PIL import Image, ImageOps


def open_upright(path: str | os.PathLike) -> Image.Image:
    """Return the image at ``path`` as RGB, turned the way its EXIF tag says."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")
