"""Compatibility shim for ``streamlit-drawable-canvas`` on Streamlit >= 1.40.

``streamlit-drawable-canvas==0.9.3`` calls
``streamlit.elements.image.image_to_url(image, width, clamp, channels,
output_format, image_id)``. Streamlit removed that symbol in 1.40 and moved the
implementation to ``streamlit.elements.lib.image_utils.image_to_url`` with a new
second argument: a ``LayoutConfig`` instead of an ``int`` width.

Importing this module re-installs a backwards-compatible ``image_to_url`` onto
``streamlit.elements.image`` that adapts the old ``width:int`` call to the new
signature. Import it *before* ``streamlit_drawable_canvas``.
"""

from __future__ import annotations

import streamlit.elements.image as _st_image


def _install() -> None:
    if hasattr(_st_image, "image_to_url"):
        return  # native (old Streamlit) — nothing to do

    from streamlit.elements.lib.image_utils import (  # noqa: PLC0415
        image_to_url as _new_image_to_url,
    )
    from streamlit.elements.lib.layout_utils import LayoutConfig  # noqa: PLC0415

    def image_to_url(  # noqa: PLR0913 - signature mirrors the old Streamlit API
        image,
        width,
        clamp,
        channels,
        output_format,
        image_id,
    ) -> str:
        layout_config = LayoutConfig(width=int(width))
        return _new_image_to_url(
            image, layout_config, clamp, channels, output_format, image_id
        )

    _st_image.image_to_url = image_to_url


_install()
