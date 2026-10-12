#!/usr/bin/env python3
"""
correlateObject.py
==================

Find a reference object (a cut-out image) inside a larger scene image by
*correlation*: the reference is laid over the scene at every position and
a range of rotations, and the place where the two agree best is taken.
The found object is then warped out of the scene so that it has exactly
the reference's size and orientation.

Why this exists next to findObject.py (SIFT)
--------------------------------------------
Feature matching needs texture. A smooth, dark, glossy part (black glass
with a logo, say) gives SIFT a dozen keypoints for the whole part: far too
few, and the few it finds are easily matched wrongly. Correlation uses
every pixel of the part -- its outline, its windows, its print -- so such
parts are found reliably. It also yields an honest quality figure (0..1),
which is used to say "not found" instead of returning a wrong alignment.

What it assumes (true for parts on the belt under the line camera):

* the object has the same size in the scene as in the reference (no
  scale search) and is not mirrored;
* it lies roughly as in the reference, or turned by half a turn (or a
  quarter turn where the reference fits that way), tilted by at most
  `max_tilt_deg`.

Steps
-----
1. Coarse search on strongly reduced copies: for each base orientation
   and each tilt, the scene is turned and the reference matched over it
   (normalised cross-correlation of the grey values and of their edges;
   both are indifferent to overall brightness, so reference and scene may
   be taken with different exposure).
2. Fine alignment on larger copies, for the best candidate of *each*
   base orientation: a search over small tilts and shifts around the
   coarse result, then OpenCV's ECC algorithm for sub-pixel accuracy.
3. The orientations are compared on their fine alignments. A part that is
   nearly symmetric (a centred window) scores almost the same both ways
   round in the coarse search; only once both are aligned precisely does
   its print or logo tell them apart.
4. Below `min_score` the object counts as not found.

Requirements: opencv-python (or -contrib), numpy.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Longest side (px) of the reference in the coarse search and in the fine
# alignment. Coarse: small enough that a search over all rotations takes
# about a second, large enough that a degree of tilt still shows. Fine: what
# the final shift and rotation are worked out on.
COARSE_REFERENCE_SIZE = 300
FINE_REFERENCE_SIZE = 1600
# Below this agreement (0..1, see _agreement) the object counts as not
# found. A reference laid over a place without the object scores around
# 0.1-0.3 on the captures this was tuned on, a found object 0.75 and more.
DEFAULT_MIN_SCORE = 0.65
# Orientations whose coarse score is this far behind the best one are not
# aligned finely, and two aligned orientations closer together than
# AMBIGUOUS_ORIENTATION_MARGIN get a warning: the part then looks nearly
# the same both ways round.
COARSE_SCORE_MARGIN = 0.15
# Share of the reference's area that has to lie inside the scene.
MIN_INSIDE = 0.9
AMBIGUOUS_ORIENTATION_MARGIN = 0.01


def _load(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _edges(gray: np.ndarray) -> np.ndarray:
    """Strength of the grey-value edges, lightly smoothed first so single
    noisy pixels (dark captures are noisy) do not count as edges."""
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    return cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))


def _resized(gray: np.ndarray, factor: float) -> np.ndarray:
    size = (max(8, round(gray.shape[1] * factor)), max(8, round(gray.shape[0] * factor)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)


def _correlation(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Normalised cross-correlation (-1..1) of two images of one size,
    over the pixels of `mask` if given. 0 where either has no contrast."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    if mask is not None:
        if mask.sum() < 16:
            return 0.0
        a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    norm = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / norm) if norm > 1e-9 else 0.0


def _agreement(reference: np.ndarray, candidate: np.ndarray, valid: np.ndarray) -> float:
    """How well `candidate` (the scene warped into the reference's frame)
    agrees with the reference, 0..1: the mean of the correlation of the
    grey values and of the edges. Grey values alone are fooled by one
    large bright area that lines up; edges alone are weak on noisy dark
    images; together they have to agree in outline *and* detail."""
    grey = _correlation(reference, candidate, valid)
    edge = _correlation(_edges(reference), _edges(candidate), valid)
    return max(0.0, (grey + edge) / 2.0)


def _placementAgreement(reference: np.ndarray, scene: np.ndarray, matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """_agreement of the reference with the scene under this placement
    (reference->scene, 3x3), and which pixels of the reference's frame lie
    inside the scene."""
    size = (reference.shape[1], reference.shape[0])
    candidate = _warpIntoReference(scene, matrix, size)
    valid = _warpIntoReference(np.full(scene.shape[:2], 255, np.uint8), matrix, size) > 0
    return _agreement(reference, candidate, valid), valid


def _referenceToScene(angleDeg: float, centre: Tuple[float, float], location: Tuple[float, float]) -> np.ndarray:
    """3x3 matrix taking reference coordinates to scene coordinates, for a
    reference found at `location` (its top left corner) in the scene after
    the scene was turned by `angleDeg` about `centre`."""
    turn = np.vstack([cv2.getRotationMatrix2D(centre, angleDeg, 1.0), [0, 0, 1]])       # scene -> turned scene
    shift = np.array([[1, 0, location[0]], [0, 1, location[1]], [0, 0, 1]], dtype=np.float64)   # reference -> turned scene
    return np.linalg.inv(turn) @ shift


def _rescaled(matrix: np.ndarray, factor: float) -> np.ndarray:
    """The same placement for images `factor` times as large."""
    out = matrix.copy()
    out[:2, 2] *= factor
    return out


def _warpIntoReference(scene: np.ndarray, matrix: np.ndarray, size: Tuple[int, int], border=cv2.BORDER_CONSTANT) -> np.ndarray:
    """The part of `scene` the reference lies on, in the reference's frame.
    `size` is (width, height) of the reference."""
    return cv2.warpAffine(
        scene, matrix[:2].astype(np.float64), size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border,
    )


def _coarseSearch(
    reference: np.ndarray, scene: np.ndarray, angles: Iterable[float],
) -> List[Tuple[float, float, np.ndarray]]:
    """Match `reference` over `scene` (both reduced, grey) for every angle.
    Returns (score, angle, reference->scene matrix) per angle, best first.

    The scene is turned rather than the reference: a turned reference
    would need a mask for its empty corners, and masked matching is many
    times slower. The scene gets a border first, so that a reference
    reaching the edge of the scene (a tightly cropped capture) still
    finds its place."""
    refH, refW = reference.shape
    pad = int(max(refH, refW) * 0.25) + 2
    # The border continues the scene's edge pixels: no artificial edge the
    # reference's outline could latch on to.
    padded = cv2.copyMakeBorder(scene, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    centre = (padded.shape[1] / 2.0, padded.shape[0] / 2.0)
    refGrey, refEdges = reference.astype(np.float32), _edges(reference)
    unpad = np.array([[1, 0, -pad], [0, 1, -pad], [0, 0, 1]], dtype=np.float64)

    results = []
    for angle in angles:
        turn = cv2.getRotationMatrix2D(centre, angle, 1.0)
        turned = cv2.warpAffine(padded, turn, (padded.shape[1], padded.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if turned.shape[0] < refH or turned.shape[1] < refW:
            continue
        scores = cv2.matchTemplate(turned.astype(np.float32), refGrey, cv2.TM_CCOEFF_NORMED)
        scores += cv2.matchTemplate(_edges(turned), refEdges, cv2.TM_CCOEFF_NORMED)
        scores[~np.isfinite(scores)] = -2.0       # flat areas divide by zero
        _, best, _, location = cv2.minMaxLoc(scores)
        results.append((best / 2.0, float(angle), unpad @ _referenceToScene(angle, centre, location)))
    results.sort(key=lambda r: r[0], reverse=True)
    return results


def _turnAboutCentre(size: Tuple[int, int], angleDeg: float) -> np.ndarray:
    """Reference coordinates -> the same, turned by `angleDeg` about the
    reference's middle. `size` is (width, height)."""
    return np.vstack([cv2.getRotationMatrix2D((size[0] / 2.0, size[1] / 2.0), angleDeg, 1.0), [0, 0, 1]])


def _refine(reference: np.ndarray, scene: np.ndarray, start: np.ndarray, tilt: float, reach: int) -> np.ndarray:
    """Improve `start` (reference->scene, 3x3) on the fine copies.

    First a plain search, which cannot get lost: for tilts of up to
    `tilt` degrees either way, the scene around the expected place is
    brought into the reference's frame with `reach` pixels to spare on
    every side, and the reference is matched inside it. Then ECC adjusts
    shift and rotation to a fraction of a pixel; if it does not settle or
    wanders off, the search's result stands."""
    refH, refW = reference.shape
    size = (refW, refH)
    refGrey, refEdges = reference.astype(np.float32), _edges(reference)
    # Into a frame `reach` larger all round: frame coordinates -> reference coordinates.
    intoFrame = np.array([[1, 0, -reach], [0, 1, -reach], [0, 0, 1]], dtype=np.float64)
    best = (-np.inf, start)
    for delta in np.linspace(-tilt, tilt, 13):
        tilted = start @ _turnAboutCentre(size, delta)
        window = _warpIntoReference(scene, tilted @ intoFrame, (refW + 2 * reach, refH + 2 * reach), border=cv2.BORDER_REPLICATE)
        scores = cv2.matchTemplate(window.astype(np.float32), refGrey, cv2.TM_CCOEFF_NORMED)
        scores += cv2.matchTemplate(_edges(window), refEdges, cv2.TM_CCOEFF_NORMED)
        scores[~np.isfinite(scores)] = -2.0
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score > best[0]:
            shift = np.array([[1, 0, x - reach], [0, 1, y - reach], [0, 0, 1]], dtype=np.float64)
            best = (score, tilted @ shift)
    searched = best[1]

    # ECC is started from a rotation near zero: it does not converge from
    # a half turn. So the reference is turned by whole quarter turns until
    # it lies like the object in the scene, and the placement is converted
    # to that turned reference and back.
    quarterTurns = int(round(math.degrees(math.atan2(searched[1, 0], searched[0, 0])) / 90.0)) % 4
    turnedReference, fromTurned = _quarterTurned(reference, quarterTurns)
    warp = (searched @ fromTurned)[:2].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6)
    try:
        _, warp = cv2.findTransformECC(turnedReference, scene, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error as e:
        logger.debug("ECC did not converge, keeping the searched alignment: %s", str(e).splitlines()[0][-90:])
        return searched
    refined = np.vstack([warp.astype(np.float64), [0, 0, 1]]) @ np.linalg.inv(fromTurned)
    # ECC only polishes: a result more than a pixel or two from the
    # search's is a different (wrong) optimum. And it maximises the
    # agreement of the grey values alone, which one large bright area can
    # dominate -- so it is only taken where it does not make the overall
    # agreement worse.
    corners = np.array([[0, 0, 1], [refW, refH, 1]], dtype=np.float64).T
    moved = np.abs((refined @ corners) - (searched @ corners))[:2].max()
    if moved > 3.0 or _placementAgreement(reference, scene, refined)[0] < _placementAgreement(reference, scene, searched)[0]:
        return searched
    return refined


def _quarterTurned(image: np.ndarray, quarterTurns: int) -> Tuple[np.ndarray, np.ndarray]:
    """`image` turned counter-clockwise (as seen on screen, y down: the
    sense of a positive rotation of a reference->scene placement) by that
    many quarter turns, and the 3x3 matrix taking coordinates of the
    turned image back to coordinates of `image`."""
    h, w = image.shape[:2]
    k = quarterTurns % 4
    if k == 0:
        return image, np.eye(3)
    if k == 2:
        return cv2.rotate(image, cv2.ROTATE_180), np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=np.float64)
    if k == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), np.array([[0, 1, 0], [-1, 0, h - 1], [0, 0, 1]], dtype=np.float64)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), np.array([[0, -1, w - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)


def findObjectByCorrelation(
    reference=None, scene=None, output: Optional[str | Path] = None, rectified_output: Optional[str | Path] = "rectified.png",
    max_tilt_deg: float = 8.0, tilt_step_deg: float = 1.0, quarter_turns: bool = True, min_score: float = DEFAULT_MIN_SCORE,
) -> Dict[str, Any]:
    """Find `reference` in `scene`; save the found object, brought to the
    reference's size and orientation, to `rectified_output` (and the scene
    with the object outlined to `output`, if given).

    Same contract as findObject.findObject: returns a dict with scale
    (always 1.0 here), rotation_deg, translation, mirrored (always False)
    and the two paths, plus "score" (0..1). Raises RuntimeError if the
    object is not found, i.e. nothing in the scene agrees with the
    reference to at least `min_score`.

    Parameters
    ----------
    max_tilt_deg, tilt_step_deg :
        The object may be tilted by up to this much against the reference
        (or against the reference turned by a half or quarter turn); the
        coarse search tries tilts this far apart.
    quarter_turns :
        Also try the reference turned by 90 and 270 degrees, where it
        fits into the scene that way.
    """
    if reference is None or scene is None:
        raise TypeError("findObjectByCorrelation() requires both 'reference' and 'scene'.")
    refBgr, sceneBgr = _load(reference), _load(scene)
    refGray = cv2.cvtColor(refBgr, cv2.COLOR_BGR2GRAY)
    sceneGray = cv2.cvtColor(sceneBgr, cv2.COLOR_BGR2GRAY)
    refH, refW = refGray.shape
    if float(refGray.std()) < 1e-3:
        raise RuntimeError("The reference image shows nothing (one flat colour).")

    # 1. coarse search -- reference and scene reduced by the same factor
    coarse = min(1.0, COARSE_REFERENCE_SIZE / float(max(refH, refW)))
    refCoarse, sceneCoarse = _resized(refGray, coarse), _resized(sceneGray, coarse)
    tilts = np.arange(-max_tilt_deg, max_tilt_deg + 1e-9, tilt_step_deg)
    bases = [0.0, 180.0] + ([90.0, 270.0] if quarter_turns else [])
    candidates = _coarseSearch(refCoarse, sceneCoarse, [base + tilt for base in bases for tilt in tilts])
    if not candidates:
        raise RuntimeError("The scene is too small to hold the reference object.")

    # 2. fine alignment of the best candidate of each base orientation
    fine = min(1.0, FINE_REFERENCE_SIZE / float(max(refH, refW)))
    refFine, sceneFine = _resized(refGray, fine), _resized(sceneGray, fine)
    # The coarse search is good to about a pixel and one tilt step of its own.
    reach = int(math.ceil(2.0 * fine / coarse)) + 2
    bestCoarse = candidates[0][0]
    aligned = []
    for base in bases:
        ofBase = [c for c in candidates if abs(c[1] - base) <= max_tilt_deg + 1e-6]
        # An orientation that is far behind in the coarse search is not
        # going to win; aligning it would only cost time.
        if not ofBase or ofBase[0][0] < bestCoarse - COARSE_SCORE_MARGIN:
            continue
        coarseScore, coarseAngle, coarseMatrix = ofBase[0]
        matrix = _refine(refFine, sceneFine, _rescaled(coarseMatrix, fine / coarse), tilt=tilt_step_deg, reach=reach)
        agreement, valid = _placementAgreement(refFine, sceneFine, matrix)
        aligned.append((agreement, matrix, valid, coarseScore, coarseAngle))
        logger.debug("orientation %3.0f: coarse %.3f at %.1f deg, aligned agreement %.3f", base, coarseScore, coarseAngle, agreement)

    # 3. the orientation that agrees best once aligned
    aligned.sort(key=lambda a: a[0], reverse=True)
    score, matrix, valid, coarseScore, coarseAngle = aligned[0]
    if len(aligned) > 1 and score >= min_score and score - aligned[1][0] < AMBIGUOUS_ORIENTATION_MARGIN:
        logger.warning(
            "The object looks almost the same in two orientations (agreement %.3f against %.3f): "
            "check that the aligned image is the right way round.", score, aligned[1][0],
        )

    # 4. found at all?
    inside = float(valid.mean())
    if score < min_score:
        raise RuntimeError(
            f"nothing in the image agrees well enough with the reference (best agreement {score:.2f}, needed {min_score:.2f})"
        )
    # A part that is cut off by the edge of the image is not handed on as
    # if it were whole (the missing strip would be made up).
    if inside < MIN_INSIDE:
        raise RuntimeError(f"only {inside * 100:.0f} % of the reference object lies inside the image")

    full = _rescaled(matrix, 1.0 / fine)
    rotationDeg = math.degrees(math.atan2(full[1, 0], full[0, 0]))
    info = {
        "scale": 1.0,
        "rotation_deg": float(rotationDeg),
        "translation": (float(full[0, 2]), float(full[1, 2])),
        "mirrored": False,
        "score": float(score),
    }
    logger.info(
        "Object found by correlation: agreement %.2f (coarse %.2f at %.1f deg), rotation %.2f deg, translation (%.1f, %.1f) px",
        score, coarseScore, coarseAngle, rotationDeg, full[0, 2], full[1, 2],
    )

    if output is not None:
        corners = np.array([[0, 0, 1], [refW, 0, 1], [refW, refH, 1], [0, refH, 1]], dtype=np.float64).T
        outline = (full @ corners)[:2].T.astype(np.int32).reshape(-1, 1, 2)
        annotated = sceneBgr.copy()
        cv2.polylines(annotated, [outline], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.imwrite(str(output), annotated)
    if rectified_output is not None:
        # Parts of the reference's frame that lie outside the scene
        # continue the scene's edge rather than turning black.
        rectified = _warpIntoReference(sceneBgr, full, (refW, refH), border=cv2.BORDER_REPLICATE)
        if not cv2.imwrite(str(rectified_output), rectified):
            raise RuntimeError(f"the found object could not be written to {rectified_output}")
    return {**info, "output": output, "rectified_output": rectified_output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find a reference object in a scene by correlation.")
    parser.add_argument("reference")
    parser.add_argument("scene")
    parser.add_argument("--output", default=None, help="Save the scene with the found object outlined here")
    parser.add_argument("--rectified-output", default="rectified.png")
    parser.add_argument("--max-tilt", type=float, default=8.0)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(findObjectByCorrelation(
        args.reference, args.scene, output=args.output, rectified_output=args.rectified_output,
        max_tilt_deg=args.max_tilt, min_score=args.min_score,
    ))
