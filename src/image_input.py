"""
==========================================
Image Input Module
==========================================

Handles image capture and upload for leaf disease
classification apps.

Crop-agnostic: this module knows nothing about rice,
apple, or any specific disease set. It only deals with
getting a clean, valid RGB image out of whatever the user
provided (a file upload or a camera photo) and handing it
off ready for the preprocessing / classification pipeline.
"""

from io import BytesIO

from PIL import Image, ImageOps
import streamlit as st

from src.config import (
    ALLOWED_IMAGE_TYPES,
    MAX_UPLOAD_SIZE_MB,
    MIN_IMAGE_DIMENSION
)


# ==========================================
# Streamlit Widgets: Upload + Camera Capture
# ==========================================

def get_image_input(label="Upload Leaf Image", key_prefix="image_input"):
    """
    Renders both an upload widget and a camera widget.

    Returns the raw file-like object the user provided
    (whichever one they used), or None if nothing was
    provided yet.
    """

    tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])

    uploaded_file = None

    with tab_upload:

        uploaded_file = st.file_uploader(

            label,

            type=ALLOWED_IMAGE_TYPES,

            key=f"{key_prefix}_uploader"

        )

    with tab_camera:

        camera_file = st.camera_input(

            "Take a photo",

            key=f"{key_prefix}_camera"

        )

        if camera_file is not None:

            uploaded_file = camera_file

    return uploaded_file


# ==========================================
# Validation + Loading
# ==========================================

def validate_and_load_image(uploaded_file):
    """
    Validates a raw uploaded/captured file and returns a
    clean, ready-to-use RGB PIL Image.

    This is the handoff boundary: anything downstream
    (preprocessing, classification) can assume the image
    it receives is a valid, correctly-oriented RGB PIL
    Image, or it will simply never receive one.

    Returns:
        (image, error_message)

        On success  : (PIL.Image, None)
        On failure  : (None, "human readable reason")
    """

    if uploaded_file is None:

        return None, "No file provided."

    # ---- Size check ----

    size_mb = uploaded_file.size / (1024 * 1024)

    if size_mb > MAX_UPLOAD_SIZE_MB:

        return None, (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum allowed is {MAX_UPLOAD_SIZE_MB} MB."
        )

    # ---- Extension check (best-effort only) ----
    # This is NOT the real security/validity gate -- a
    # renamed file can still pass this check. The content
    # verification below is what actually matters.

    file_name = getattr(uploaded_file, "name", "capture.jpg")

    extension = (
        file_name.rsplit(".", 1)[-1].lower()
        if "." in file_name
        else ""
    )

    if extension and extension not in ALLOWED_IMAGE_TYPES:

        return None, (
            f"Unsupported file type '.{extension}'. "
            f"Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}."
        )

    # ---- Real content verification ----

    try:

        raw_bytes = uploaded_file.getvalue()

        # verify() checks file integrity but leaves the
        # image unusable afterwards, so we reopen it
        image = Image.open(BytesIO(raw_bytes))
        image.verify()

        image = Image.open(BytesIO(raw_bytes))

    except Exception:

        return None, (
            "This file could not be read as a valid image. "
            "Please try a different file."
        )

    # ---- EXIF orientation fix ----
    # Phone photos often carry an EXIF orientation tag, so
    # the raw pixel data can be sideways even though the
    # photo looks upright on the phone. Fix this before
    # anything downstream (classification, Grad-CAM) sees it.

    try:

        image = ImageOps.exif_transpose(image)

    except Exception:

        pass  # not all images carry EXIF data; safe to skip

    # ---- Normalize to RGB ----
    # Handles grayscale, CMYK, and RGBA (e.g. PNGs with
    # transparency) inputs consistently.

    image = image.convert("RGB")

    # ---- Minimum dimension check ----

    if image.width < MIN_IMAGE_DIMENSION or image.height < MIN_IMAGE_DIMENSION:

        return None, (
            f"Image too small ({image.width}x{image.height}). "
            f"Minimum required is "
            f"{MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}."
        )

    return image, None