import os, json, time, uuid, shutil, threading
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE     = Path("/app/maps")
BASE.mkdir(exist_ok=True)
STATUS   = {}

# ── Try importing HLoc ───────────────────────────────────────────────────────
HLOC_OK = False
try:
    from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
    from hloc.utils import read_write_model
    import pycolmap
    HLOC_OK = True
    print("✅ HLoc loaded successfully")
except Exception as e:
    print(f"⚠️ HLoc not available: {e}")

# ── Feature configs ──────────────────────────────────────────────────────────
FEATURE_CONF  = extract_features.confs['superpoint_aachen'] if HLOC_OK else None
MATCHER_CONF  = match_features.confs['superglue']           if HLOC_OK else None

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    maps = [d.name for d in BASE.iterdir() if d.is_dir()] if BASE.exists() else []
    return {
        "status": "AeroGuide HLoc running ✅",
        "hloc":   "✅ loaded" if HLOC_OK else "⚠️ not loaded",
        "maps":   maps,
        "config": "fps=1, max_frames=30"
    }

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD & PROCESS VIDEO → HLoc map + SLAM 2D map
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    video: UploadFile = File(...),
    building_id: str = Form("1"),
    floor: str = Form("1")
):
    map_id  = f"map_{uuid.uuid4().hex[:13]}"
    map_dir = BASE / map_id
    map_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video
    video_path = map_dir / "video.mp4"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Save metadata
    (map_dir / "meta.json").write_text(json.dumps({
        "building_id": building_id,
        "floor": floor,
        "map_id": map_id
    }))

    update_status(map_id, "extracting_frames", "Extracting frames...")

    # Process in background
    threading.Thread(target=process_map, args=(map_id, map_dir, video_path), daemon=True).start()

    return {"success": True, "map_id": map_id}

# ─────────────────────────────────────────────────────────────────────────────
def process_map(map_id, map_dir, video_path):
    try:
        imgs_dir = map_dir / "images"
        imgs_dir.mkdir(exist_ok=True)

        # ── Extract frames ───────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps))
        frame_idx, saved = 0, 0
        MAX_FRAMES = 30

        frame_positions = []  # For SLAM 2D map

        while saved < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                path = imgs_dir / f"frame_{saved:04d}.jpg"
                h, w = frame.shape[:2]
                scale = 640 / max(w, h)
                frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
                cv2.imwrite(str(path), frame_small)
                frame_positions.append(saved)
                saved += 1
            frame_idx += 1
        cap.release()

        if saved < 5:
            update_status(map_id, "failed", f"Only {saved} frames extracted")
            return

        update_status(map_id, "running_hloc", f"Running HLoc on {saved} frames...")

        if HLOC_OK:
            run_hloc(map_id, map_dir, imgs_dir, saved)
        else:
            # Fallback: build basic SLAM map from optical flow
            run_slam_fallback(map_id, map_dir, imgs_dir, saved)

    except Exception as e:
        update_status(map_id, "failed", str(e))

# ─────────────────────────────────────────────────────────────────────────────
def run_hloc(map_id, map_dir, imgs_dir, n_frames):
    try:
        sfm_dir   = map_dir / "sfm"
        pairs_f   = map_dir / "pairs.txt"
        feats_f   = map_dir / "features.h5"
        matches_f = map_dir / "matches.h5"

        sfm_dir.mkdir(exist_ok=True)

        # Generate pairs
        refs = [f"frame_{i:04d}.jpg" for i in range(n_frames)]
        pairs_from_exhaustive.main(pairs_f, image_list=refs)

        # Extract features
        extract_features.main(FEATURE_CONF, imgs_dir, image_list=refs, feature_path=feats_f)

        # Match features
        match_features.main(MATCHER_CONF, pairs_f, features=feats_f, matches=matches_f)

        # COLMAP reconstruction
        model = reconstruction.main(
            sfm_dir, imgs_dir, pairs_f, feats_f, matches_f,
            verbose=False
        )

        if model is None or len(model.images) < 3:
            update_status(map_id, "failed", "COLMAP reconstruction failed")
            return

        # Extract camera positions → build 2D SLAM map
        slam_map = extract_slam_map(model, map_id)
        (map_dir / "slam_map.json").write_text(json.dumps(slam_map))

        update_status(map_id, "ready", f"✅ Map ready! {len(model.images)} frames reconstructed")

    except Exception as e:
        update_status(map_id, "failed", f"HLoc error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
def extract_slam_map(model, map_id):
    """
    Extract 2D GTA-style map from COLMAP reconstruction.
    Returns walls, path, and scale for frontend rendering.
    """
    positions = []
    for img_id, img in model.images.items():
        # Camera position in world coordinates
        R = img.rotmat()
        t = img.tvec
        pos = -R.T @ t
        positions.append({'x': float(pos[0]), 'z': float(pos[2])})

    if not positions:
        return {}

    xs = [p['x'] for p in positions]
    zs = [p['z'] for p in positions]

    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    # Normalize to 0-500 canvas
    W, H = 500, 500
    pad = 40

    def norm_x(x): return int(pad + (x - min_x) / max(max_x - min_x, 0.001) * (W - pad*2))
    def norm_z(z): return int(pad + (z - min_z) / max(max_z - min_z, 0.001) * (H - pad*2))

    # Path points (camera trajectory)
    path = [{'x': norm_x(p['x']), 'z': norm_z(p['z'])} for p in positions]

    # Build walls from trajectory (simplified: corridor detection)
    walls = []
    corridor_width = 20  # pixels

    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        dx = p2['x'] - p1['x']
        dz = p2['z'] - p1['z']
        length = max((dx**2 + dz**2)**0.5, 0.001)
        nx = -dz / length * corridor_width
        nz = dx / length * corridor_width

        walls.append({
            'x1': p1['x'] + nx, 'z1': p1['z'] + nz,
            'x2': p2['x'] + nx, 'z2': p2['z'] + nz
        })
        walls.append({
            'x1': p1['x'] - nx, 'z1': p1['z'] - nz,
            'x2': p2['x'] - nx, 'z2': p2['z'] - nz
        })

    # Scale factor: real world units per pixel
    real_width  = max_x - min_x
    real_height = max_z - min_z
    scale = max(real_width, real_height) / (W - pad*2) if (W - pad*2) > 0 else 1

    return {
        'path':     path,
        'walls':    walls,
        'min_x':    min_x,
        'min_z':    min_z,
        'max_x':    max_x,
        'max_z':    max_z,
        'scale':    scale,
        'canvas_w': W,
        'canvas_h': H,
        'pad':      pad
    }

# ─────────────────────────────────────────────────────────────────────────────
def run_slam_fallback(map_id, map_dir, imgs_dir, n_frames):
    """
    Fallback SLAM using OpenCV optical flow when HLoc not available.
    Builds approximate 2D map from camera motion estimation.
    """
    update_status(map_id, "running_hloc", "Running optical flow SLAM...")

    frames = sorted(imgs_dir.glob("*.jpg"))
    if len(frames) < 3:
        update_status(map_id, "failed", "Not enough frames")
        return

    # Track camera motion using optical flow
    prev_gray = cv2.cvtColor(cv2.imread(str(frames[0])), cv2.COLOR_BGR2GRAY)
    
    positions = [{'x': 0.0, 'z': 0.0}]
    x, z = 0.0, 0.0
    angle = 0.0

    lk_params = dict(winSize=(21,21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    for frame_path in frames[1:min(n_frames, len(frames))]:
        curr_gray = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2GRAY)

        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
        if prev_pts is None or len(prev_pts) < 10:
            prev_gray = curr_gray
            positions.append({'x': x, 'z': z})
            continue

        # Track features
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 5:
            prev_gray = curr_gray
            positions.append({'x': x, 'z': z})
            continue

        # Estimate camera motion
        try:
            E, mask = cv2.findEssentialMat(good_curr, good_prev,
                                            focal=700, pp=(curr_gray.shape[1]//2, curr_gray.shape[0]//2),
                                            method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, good_curr, good_prev,
                                              focal=700, pp=(curr_gray.shape[1]//2, curr_gray.shape[0]//2))
                # Accumulate translation
                x += float(t[0][0]) * 0.5
                z += float(t[2][0]) * 0.5
        except:
            pass

        positions.append({'x': x, 'z': z})
        prev_gray = curr_gray

    # Build SLAM map from positions
    if not positions:
        update_status(map_id, "failed", "SLAM failed")
        return

    xs = [p['x'] for p in positions]
    zs = [p['z'] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    W, H, pad = 500, 500, 40

    def norm_x(x): return int(pad + (x - min_x) / max(max_x - min_x, 0.001) * (W - pad*2))
    def norm_z(z): return int(pad + (z - min_z) / max(max_z - min_z, 0.001) * (H - pad*2))

    path = [{'x': norm_x(p['x']), 'z': norm_z(p['z'])} for p in positions]

    # Build corridor walls
    walls = []
    cw = 18
    for i in range(len(path)-1):
        p1, p2 = path[i], path[i+1]
        dx = p2['x'] - p1['x']
        dz = p2['z'] - p1['z']
        length = max((dx**2+dz**2)**0.5, 0.001)
        nx, nz = -dz/length*cw, dx/length*cw
        walls.append({'x1':p1['x']+nx,'z1':p1['z']+nz,'x2':p2['x']+nx,'z2':p2['z']+nz})
        walls.append({'x1':p1['x']-nx,'z1':p1['z']-nz,'x2':p2['x']-nx,'z2':p2['z']-nz})

    scale = max(max_x-min_x, max_z-min_z) / (W-pad*2) if (W-pad*2) > 0 else 1

    slam_map = {
        'path': path, 'walls': walls,
        'min_x': min_x, 'min_z': min_z,
        'max_x': max_x, 'max_z': max_z,
        'scale': scale, 'canvas_w': W, 'canvas_h': H, 'pad': pad
    }
    (map_dir / "slam_map.json").write_text(json.dumps(slam_map))
    update_status(map_id, "ready", f"✅ Map ready (optical flow SLAM, {len(positions)} positions)")

# ─────────────────────────────────────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status/{map_id}")
def get_status(map_id: str):
    if map_id in STATUS:
        return {**STATUS[map_id], "map_id": map_id}
    
    map_dir = BASE / map_id
    if map_dir.exists():
        slam = (map_dir / "slam_map.json").exists()
        return {"map_id": map_id, "status": "ready", "message": "✅ Map ready", "has_slam": slam}
    
    return {"map_id": map_id, "status": "not_found", "message": "Map not found"}

# ─────────────────────────────────────────────────────────────────────────────
# GET SLAM MAP
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/slam/{map_id}")
def get_slam(map_id: str):
    slam_path = BASE / map_id / "slam_map.json"
    if not slam_path.exists():
        raise HTTPException(404, "SLAM map not found")
    return json.loads(slam_path.read_text())

# ─────────────────────────────────────────────────────────────────────────────
# LOCALIZE — find position from image
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/localize")
async def localize(
    image: UploadFile = File(...),
    building_id: str = Form("1"),
    floor: str = Form("1")
):
    # Find latest ready map
    ready_maps = []
    if BASE.exists():
        for d in BASE.iterdir():
            if d.is_dir() and (d/"slam_map.json").exists():
                ready_maps.append(d)

    if not ready_maps:
        return demo_position("no_maps")

    map_dir = sorted(ready_maps, key=lambda d: d.stat().st_mtime)[-1]

    if not HLOC_OK:
        return demo_position("hloc_not_loaded")

    # Save query image
    img_data = await image.read()
    query_path = map_dir / "query_tmp.jpg"
    query_path.write_bytes(img_data)

    try:
        result = run_hloc_localize(map_dir, query_path)
        query_path.unlink(missing_ok=True)
        return result
    except Exception as e:
        query_path.unlink(missing_ok=True)
        return demo_position(f"localize_error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
def run_hloc_localize(map_dir, query_path):
    sfm_dir = map_dir / "sfm"
    if not sfm_dir.exists():
        return demo_position("no_sfm")

    try:
        model = pycolmap.Reconstruction(str(sfm_dir))
        
        # Use COLMAP image_registrator for localization
        camera = pycolmap.Camera(
            model='SIMPLE_RADIAL',
            width=640, height=480,
            params=[700, 320, 240, 0]
        )

        query_img = {
            'image': str(query_path),
            'camera': camera
        }

        # Extract features for query
        query_feats = map_dir / "query_feats.h5"
        extract_features.main(
            FEATURE_CONF,
            query_path.parent,
            image_list=[query_path.name],
            feature_path=query_feats
        )

        # Match against database
        db_feats  = map_dir / "features.h5"
        db_pairs  = map_dir / "pairs.txt"
        db_matches = map_dir / "matches.h5"

        # Get best reference images
        pairs_from_exhaustive.main(
            map_dir / "query_pairs.txt",
            image_list=[query_path.name],
            ref_list=[f"frame_{i:04d}.jpg" for i in range(min(20, len(list(model.images))))]
        )

        match_features.main(
            MATCHER_CONF,
            map_dir / "query_pairs.txt",
            features=db_feats,
            matches=map_dir / "query_matches.h5",
            features_ref=query_feats
        )

        # Localize
        pose = pycolmap.absolute_pose_estimation(
            str(query_path), model, camera
        )

        if pose and pose.success:
            pos = pose.cam_from_world.inverse().translation
            return {
                "success": True,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "confidence": 0.9,
                "demo": False
            }
        else:
            return demo_position("pose_estimation_failed")

    except Exception as e:
        return demo_position(f"hloc_localize: {e}")

# ─────────────────────────────────────────────────────────────────────────────
def demo_position(reason="demo"):
    """
    Returns demo position with LOW confidence (< 0.5)
    so frontend knows to IGNORE it.
    """
    import random
    return {
        "success": True,
        "x": random.uniform(0, 15),
        "y": 0,
        "z": random.uniform(0, 25),
        "confidence": 0.3,   # ← LOW: frontend should ignore
        "demo": True,
        "reason": reason
    }

# ─────────────────────────────────────────────────────────────────────────────
def update_status(map_id, status, message=""):
    STATUS[map_id] = {"status": status, "message": message}
    print(f"[{map_id}] {status}: {message}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
