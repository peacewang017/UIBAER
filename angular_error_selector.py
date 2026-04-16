"""
Angular Reproduction Error (ARE) Calculator
============================================
Interactively select 6 blocks in an image, compute the Angular Reproduction
Error for each, and report the mean.

Usage:
    python angular_error_selector.py <image_path>

Controls (inside the OpenCV window):
    Left-click + drag  – draw a selection rectangle
    ENTER / SPACE      – confirm the current selection and move to next block
    R                  – reset / redraw the current block's selection
    ESC                – quit early
"""

import sys
import cv2
import numpy as np


# ─────────────────────────────── helpers ────────────────────────────────── #

def angular_reproduction_error(patch: np.ndarray) -> float:
    """
    Compute the Angular Reproduction Error (ARE) for a single patch.

    ARE measures how much the colour direction of each pixel deviates from
    the mean colour direction of the patch.

        ARE = (1/N) * Σ  arccos( p̂ᵢ · μ̂ )

    where p̂ᵢ is the unit-vector of pixel i in RGB space and μ̂ is the
    unit-vector of the patch mean colour.  The result is in degrees.

    Parameters
    ----------
    patch : np.ndarray
        H×W×3 uint8 BGR image patch.

    Returns
    -------
    float
        Mean angular error in degrees.
    """
    # Work in float; convert BGR → RGB (direction only, so order matters for
    # semantic correctness but not for the magnitude of the error).
    rgb = patch.reshape(-1, 3).astype(np.float64)

    # Remove near-black pixels to avoid division-by-zero artefacts.
    norms = np.linalg.norm(rgb, axis=1)
    valid = norms > 1e-6
    if valid.sum() == 0:
        return 0.0

    rgb = rgb[valid]
    norms = norms[valid]

    # Unit vectors for each pixel.
    unit_pixels = rgb / norms[:, np.newaxis]          # (N, 3)

    # Mean colour direction.
    mean_color = rgb.mean(axis=0)
    mean_norm = np.linalg.norm(mean_color)
    if mean_norm < 1e-6:
        return 0.0
    unit_mean = mean_color / mean_norm                 # (3,)

    # Dot products clamped to [-1, 1] to guard against float rounding.
    dots = np.clip(unit_pixels @ unit_mean, -1.0, 1.0)

    # Angular error per pixel (radians → degrees).
    angles = np.degrees(np.arccos(dots))
    return float(angles.mean())


# ─────────────────────────────── GUI state ──────────────────────────────── #

class ROISelector:
    """Manages mouse-driven rectangular selection on an OpenCV window."""

    def __init__(self):
        self.start = None
        self.end = None
        self.drawing = False
        self.confirmed = False

    def reset(self):
        self.start = None
        self.end = None
        self.drawing = False
        self.confirmed = False

    @property
    def rect(self):
        """Return (x, y, w, h) or None."""
        if self.start is None or self.end is None:
            return None
        x0, y0 = self.start
        x1, y1 = self.end
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 2 or h < 2:
            return None
        return (x, y, w, h)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.confirmed = False
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)


# ──────────────────────────── display helpers ────────────────────────────── #

PALETTE = [
    (0,   200, 255),   # yellow-ish
    (50,  255, 50),    # green
    (255, 100, 50),    # blue-ish
    (0,   120, 255),   # orange
    (200, 50,  255),   # purple
    (255, 50,  150),   # pink
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
THICKNESS  = 2
N_BLOCKS   = 6

def draw_ui(base: np.ndarray, selector: ROISelector, block_idx: int,
            results: list) -> np.ndarray:
    """Compose the display frame from the base image + overlays."""
    canvas = base.copy()
    h, w   = canvas.shape[:2]

    # ── already-confirmed selections ──────────────────────────────────────
    for i, (rect, are) in enumerate(results):
        x, y, bw, bh = rect
        color = PALETTE[i % len(PALETTE)]
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
        label = f"B{i+1}: {are:.2f} deg"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE * 0.85, 1)
        # Small filled box behind label for legibility.
        cv2.rectangle(canvas, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(canvas, label, (x + 2, y - 4),
                    FONT, FONT_SCALE * 0.85, (0, 0, 0), 1, cv2.LINE_AA)

    # ── current live selection ─────────────────────────────────────────────
    rect = selector.rect
    if rect:
        x, y, bw, bh = rect
        color = PALETTE[block_idx % len(PALETTE)]
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh),
                      tuple(c // 3 for c in color), -1)   # dim fill

    # ── HUD banner ────────────────────────────────────────────────────────
    banner_h = 42
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (20, 20, 20), -1)

    if block_idx < N_BLOCKS:
        msg = (f"Block {block_idx + 1}/{N_BLOCKS} — "
               "drag to select  |  ENTER=confirm  R=reset  ESC=quit")
    else:
        mean_are = np.mean([r for _, r in results])
        msg = (f"Done!  Mean ARE = {mean_are:.4f} deg  "
               "(press ESC or any key to exit)")

    cv2.putText(canvas, msg, (10, 28), FONT, FONT_SCALE,
                (220, 220, 220), THICKNESS - 1, cv2.LINE_AA)

    return canvas


# ──────────────────────────────── main ──────────────────────────────────── #

def main():
    if len(sys.argv) < 2:
        print("Usage: python angular_error_selector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not open '{image_path}'")
        sys.exit(1)

    WIN = "ARE Calculator"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(image.shape[1], 1400),
                         min(image.shape[0],  900))

    selector = ROISelector()
    cv2.setMouseCallback(WIN, selector.mouse_cb)

    results    = []   # list of (rect, are_value)
    block_idx  = 0

    print("\nAngular Reproduction Error Calculator")
    print("=" * 40)
    print(f"Image : {image_path}  ({image.shape[1]}×{image.shape[0]})")
    print(f"Blocks: {N_BLOCKS}")
    print()

    while True:
        frame = draw_ui(image, selector, block_idx, results)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(30) & 0xFF

        # ── ESC ─────────────────────────────────────────────────────────
        if key == 27:
            print("\nAborted by user.")
            break

        # ── All blocks done ─────────────────────────────────────────────
        if block_idx >= N_BLOCKS:
            # Just show results; any key (other than ESC) also exits.
            if key != 255:
                break
            continue

        # ── R – reset current selection ─────────────────────────────────
        if key == ord('r') or key == ord('R'):
            selector.reset()
            print(f"  Block {block_idx + 1}: selection reset.")

        # ── ENTER / SPACE – confirm ──────────────────────────────────────
        elif key in (13, ord(' ')):
            rect = selector.rect
            if rect is None:
                print(f"  Block {block_idx + 1}: no valid selection yet — "
                      "draw a rectangle first.")
                continue

            x, y, bw, bh = rect
            patch = image[y:y + bh, x:x + bw]
            are   = angular_reproduction_error(patch)

            results.append((rect, are))
            print(f"  Block {block_idx + 1}: x={x} y={y} w={bw} h={bh}  "
                  f"→  ARE = {are:.4f}°")

            block_idx += 1
            selector.reset()

            if block_idx == N_BLOCKS:
                mean_are = np.mean([r for _, r in results])
                print()
                print("=" * 40)
                print(f"Individual AREs : "
                      + "  ".join(f"{r:.4f}°" for _, r in results))
                print(f"Mean ARE        : {mean_are:.4f}°")
                print("=" * 40)
                print("\nPress any key in the window to exit.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
