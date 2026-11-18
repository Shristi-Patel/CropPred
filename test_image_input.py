"""
Quick sanity test for src/image_input.py's validate_and_load_image().
Not a formal pytest suite -- just exercises the edge cases discussed
before wiring this into the real app.
"""

import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image
from src.image_input import validate_and_load_image


class FakeUploadedFile:
    """Mimics the parts of Streamlit's UploadedFile our function uses."""

    def __init__(self, name, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)

    def getvalue(self):
        return self._data


def make_image_bytes(width, height, fmt="JPEG", color=(120, 160, 90)):
    buf = BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format=fmt)
    return buf.getvalue()


results = []


def run_case(label, file_obj):
    image, error = validate_and_load_image(file_obj)
    if error:
        results.append((label, "REJECTED", error))
    else:
        results.append((label, "ACCEPTED", f"{image.size[0]}x{image.size[1]} {image.mode}"))


# 1. Real sample photo (phone camera image, exercises EXIF path)
real_photo_path = Path(__file__).resolve().parent / "test_images" / "IMG_20190419_095429.jpg"

if real_photo_path.exists():
    with open(real_photo_path, "rb") as f:
        real_bytes = f.read()
    run_case("Real phone photo (IMG_20190419_095429.jpg)", FakeUploadedFile("IMG_20190419_095429.jpg", real_bytes))
else:
    results.append(("Real phone photo (IMG_20190419_095429.jpg)", "SKIPPED", f"File not found at {real_photo_path}"))

# 2. Valid, normal synthetic image
run_case("Valid 300x300 PNG", FakeUploadedFile("leaf.png", make_image_bytes(300, 300, "PNG")))

# 3. Too small
run_case("Too small (20x20)", FakeUploadedFile("tiny.jpg", make_image_bytes(20, 20)))

# 4. Text file renamed to .jpg (fake extension)
run_case("Text file renamed .jpg", FakeUploadedFile("fake.jpg", b"this is not an image, just text pretending to be one" * 5))

# 5. Disallowed extension
run_case("Disallowed extension (.gif)", FakeUploadedFile("leaf.gif", make_image_bytes(200, 200, "PNG")))

# 6. Oversized file (fake, just padding bytes past a valid image to blow past size cap)
big_image = make_image_bytes(500, 500)
padded = big_image + b"\x00" * (11 * 1024 * 1024)  # pad past 10 MB cap
run_case("Oversized file (~11MB)", FakeUploadedFile("huge.jpg", padded))

# 7. Grayscale image (mode conversion path)
buf = BytesIO()
Image.new("L", (300, 300), color=128).save(buf, format="PNG")
run_case("Grayscale PNG", FakeUploadedFile("gray.png", buf.getvalue()))

# 8. No file provided
run_case("No file provided", None)


print(f"{'CASE':45} {'RESULT':10} DETAIL")
print("-" * 100)
for label, outcome, detail in results:
    print(f"{label:45} {outcome:10} {detail}")