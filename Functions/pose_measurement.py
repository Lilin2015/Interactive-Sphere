"""Pose measurement and pose visualization."""

import os
import time

import cv2
import numpy as np


# Default intrinsics of the bundled calibration, K stored transposed in the .mat
_HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(_HERE, "..", "Material", "K_camera.mat")


def load_calib(path=CALIB_PATH):
    """Load (K, dist) from a .mat file; None if missing."""
    if not os.path.exists(path):
        return None
    import scipy.io as sio
    m = sio.loadmat(path)
    K = np.asarray(m["K"], dtype=float).T
    rad = np.asarray(m["Rad"], dtype=float).ravel()
    dist = np.array([rad[0], rad[1], 0.0, 0.0, 0.0])  # radial k1, k2 only
    return K, dist


def estimate_pose(sd, result, calib, prof=None):
    """PnP RANSAC (SQPNP, 3px) + LM refinement (Motion-only BA, point
    cloud fixed). Returns None with <6 correspondences or <6 inliers."""
    if calib is None or not result.get("accepted"):
        return None
    K, dist = calib
    pts2d, pts3d = [], []
    for (x, y), did in result["map"].items():
        if did in sd.xyz:
            pts2d.append((x, y))
            pts3d.append(sd.xyz[did])
    if len(pts2d) < 6:
        return None
    img = np.asarray(pts2d, dtype=np.float64).reshape(-1, 1, 2)
    obj = np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3)
    _t = time.perf_counter()
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, dist, iterationsCount=300,
        reprojectionError=3.0, flags=cv2.SOLVEPNP_SQPNP)
    if prof is not None:
        prof["pnp_ransac"] = (prof.get("pnp_ransac", 0.0)
                              + time.perf_counter() - _t)
    if not ok or inl is None or len(inl) < 6:
        return None
    inl_mask = np.zeros(len(pts2d), dtype=bool)
    inl_mask[inl[:, 0]] = True
    _t = time.perf_counter()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl[:, 0]], img[inl[:, 0]],
                                      K, dist, rvec, tvec)
    if prof is not None:
        prof["pnp_refine"] = (prof.get("pnp_refine", 0.0)
                              + time.perf_counter() - _t)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj = proj.reshape(-1, 2)
    err = np.linalg.norm(reproj[inl_mask] - img[inl_mask].reshape(-1, 2),
                         axis=1)
    return {"rvec": rvec, "tvec": tvec,
            "pts2d": np.asarray(pts2d, dtype=np.float64),
            "inl": inl_mask, "reproj": reproj,
            "n_in": int(inl_mask.sum()), "n_total": len(pts2d),
            "med_err": float(np.median(err))}


def draw_model_view(img, sd, pose, calib, scale=1.0):
    """Overlay the model wireframe at the current pose; back-facing edges
    are hidden, pentagon centers are magenta."""
    K, dist = calib
    ids = sorted(sd.xyz)
    pts = np.array([sd.xyz[i] for i in ids], dtype=np.float64)
    R, _ = cv2.Rodrigues(pose["rvec"])
    proj, _ = cv2.projectPoints(
        pts.reshape(-1, 1, 3), pose["rvec"], pose["tvec"], K, dist)
    proj = proj.reshape(-1, 2) * scale
    pc = (R @ pts.T).T + pose["tvec"].ravel()   # camera-frame coordinates
    nrm = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    facing = (R @ nrm.T).T                       # camera-frame normals
    front = np.einsum("ij,ij->i", facing, pc) < 0  # normal faces the camera
    pos = {did: proj[k] for k, did in enumerate(ids)}
    vis = {did: front[k] for k, did in enumerate(ids)}
    for a, b in sd.edge_list:
        if not (vis[a] and vis[b]):
            continue
        pa, pb = pos[a], pos[b]
        cv2.line(img, (int(pa[0]), int(pa[1])),
                 (int(pb[0]), int(pb[1])), (0, 255, 0), 1, cv2.LINE_AA)
    t = pose["tvec"].ravel()
    for c in sd.pent_centers:
        p3 = R @ c + t
        if (R @ (c / np.linalg.norm(c))) @ p3 >= 0:
            continue  # back-facing
        q, _ = cv2.projectPoints(c.reshape(1, 1, 3), pose["rvec"],
                                 pose["tvec"], K, dist)
        x, y = q.reshape(2) * scale
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), -1,
                   cv2.LINE_AA)


def proj_diameter(sd, pose, calib):
    """Projected sphere diameter (pixels): 95th percentile of distances
    to the projected centroid, times 2 (robust, used by Auto Set)."""
    K, dist = calib
    ids = sorted(sd.xyz)
    pts = np.array([sd.xyz[i] for i in ids], dtype=np.float64)
    proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), pose["rvec"],
                                pose["tvec"], K, dist)
    proj = proj.reshape(-1, 2)
    d = np.linalg.norm(proj - proj.mean(0), axis=1)
    return float(np.percentile(d, 95) * 2)
