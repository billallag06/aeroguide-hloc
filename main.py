"""
AeroGuide HLoc Server — Railway
Simple version: HLoc only, no SLAM, MAX_FRAMES=25
+ demo_position confidence < 0.5 so frontend ignores it
"""
import os, shutil, tempfile, asyncio, subprocess, uuid, json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = Path("/app/maps")
BASE.mkdir(exist_ok=True)

# Config
MAX_FRAMES = 25   # ✅ Léger et rapide
FPS        = 1    # 1 frame/sec

# ── Try HLoc ────────────────────────────────────────────────────────────────
HLOC_OK = False
try:
    from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
    import pycolmap
    HLOC_OK = True
    print("✅ HLoc loaded")
except Exception as e:
    print(f"⚠️ HLoc not available: {e}")

# ── HEALTH ───────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    maps = [d.name for d in BASE.iterdir() if d.is_dir()] if BASE.exists() else []
    return {
        "status": "AeroGuide HLoc running ✅",
        "hloc":   "✅ loaded" if HLOC_OK else "⚠️ not loaded",
        "maps":   maps,
        "ffmpeg": shutil.which("ffmpeg") or "not found",
        "config": f"fps={FPS}, max_frames={MAX_FRAMES}"
    }

# ── UPLOAD ───────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    video:       UploadFile = File(...),
    building_id: str        = Form("1"),
    floor:       str        = Form("1"),
    map_id:      str        = Form(None)
):
    mid     = map_id or f"map_{uuid.uuid4().hex[:13]}"
    map_dir = BASE / mid
    map_dir.mkdir(parents=True, exist_ok=True)

    video_path = map_dir / "video.mp4"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    (map_dir / "meta.json").write_text(json.dumps({
        "building_id": building_id, "floor": floor, "map_id": mid
    }))

    update_status(mid, "extracting_frames", "Extracting frames...")
    asyncio.create_task(process_map(mid, map_dir, video_path))

    return {"success": True, "map_id": mid, "status": "processing"}

# ── PROCESS MAP ──────────────────────────────────────────────────────────────
async def process_map(map_id, map_dir, video_path):
    try:
        imgs_dir = map_dir / "images"
        imgs_dir.mkdir(exist_ok=True)

        # Extract frames with OpenCV (no ffmpeg needed)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps / FPS))
        frame_idx, saved = 0, 0

        while saved < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % interval == 0:
                path = imgs_dir / f"frame_{saved:04d}.jpg"
                # Resize to 640x480 for speed
                h, w = frame.shape[:2]
                scale = 640 / max(w, h)
                small = cv2.resize(frame, (int(w*scale), int(h*scale)))
                cv2.imwrite(str(path), small)
                saved += 1
            frame_idx += 1
        cap.release()

        print(f"[{map_id}] Extracted {saved} frames", flush=True)

        if saved < 5:
            update_status(map_id, "failed", f"Only {saved} frames extracted")
            return

        update_status(map_id, "running_hloc", f"Running HLoc on {saved} frames...")

        if HLOC_OK:
            await run_hloc(map_id, map_dir, imgs_dir)
        else:
            print(f"[{map_id}] HLoc not available — marking ready anyway", flush=True)
            update_status(map_id, "ready", "Ready (no HLoc — demo mode)")

    except Exception as e:
        print(f"[{map_id}] Error: {e}", flush=True)
        update_status(map_id, "failed", str(e)[:120])

# ── RUN HLOC ─────────────────────────────────────────────────────────────────
async def run_hloc(map_id, map_dir, imgs_dir):
    try:
        sfm_dir   = map_dir / "sfm"
        pairs_f   = map_dir / "pairs.txt"
        feats_f   = map_dir / "features.h5"
        matches_f = map_dir / "matches.h5"
        sfm_dir.mkdir(exist_ok=True)

        img_list = [f.name for f in sorted(imgs_dir.glob("*.jpg"))]

        # Feature extraction
        print(f"[{map_id}] Extracting SuperPoint features...", flush=True)
        extract_features.main(
            extract_features.confs["superpoint_aachen"],
            imgs_dir,
            image_list=img_list,
            feature_path=feats_f
        )

        # Exhaustive pairs
        pairs_from_exhaustive.main(pairs_f, image_list=img_list)

        # Matching
        print(f"[{map_id}] SuperGlue matching...", flush=True)
        match_features.main(
            match_features.confs["superglue"],
            pairs_f,
            features=feats_f,
            matches=matches_f
        )

        # Reconstruction
        print(f"[{map_id}] COLMAP reconstruction...", flush=True)
        model = reconstruction.main(
            sfm_dir, imgs_dir, pairs_f, feats_f, matches_f,
            verbose=False
        )

        if model and len(model.images) >= 3:
            update_status(map_id, "ready", f"✅ Map ready! {len(model.images)} frames reconstructed")
            print(f"[{map_id}] ✅ Ready!", flush=True)
        else:
            update_status(map_id, "failed", "COLMAP reconstruction failed — too few images registered")

    except Exception as e:
        print(f"[{map_id}] HLoc error: {e}", flush=True)
        update_status(map_id, "failed", f"HLoc error: {str(e)[:100]}")

# ── STATUS ───────────────────────────────────────────────────────────────────
@app.get("/status/{map_id}")
def get_status(map_id: str):
    if map_id in STATUS:
        return {**STATUS[map_id], "map_id": map_id}
    map_dir = BASE / map_id
    if map_dir.exists():
        return {"map_id": map_id, "status": "ready", "message": "Map ready ✅"}
    return {"map_id": map_id, "status": "not_found", "message": "Map not found"}

# ── LOCALIZE ─────────────────────────────────────────────────────────────────
@app.post("/localize")
async def localize(
    image:       UploadFile = File(...),
    building_id: str        = Form("1"),
    floor:       str        = Form("1")
):
    # Find latest ready map
    ready = []
    if BASE.exists():
        for d in BASE.iterdir():
            s = STATUS.get(d.name, {}).get("status", "")
            if d.is_dir() and (s == "ready" or (d/"sfm").exists()):
                ready.append(d)

    if not ready:
        return demo_position("no_maps")

    map_dir = sorted(ready, key=lambda d: d.stat().st_mtime)[-1]

    if not HLOC_OK:
        return demo_position("hloc_not_loaded")

    # Save query image
    img_data   = await image.read()
    query_path = map_dir / "query_tmp.jpg"
    query_path.write_bytes(img_data)

    try:
        result = run_hloc_localize(map_dir, query_path)
        return result
    except Exception as e:
        return demo_position(f"localize_error: {e}")
    finally:
        query_path.unlink(missing_ok=True)

def run_hloc_localize(map_dir, query_path):
    sfm_dir = map_dir / "sfm"
    if not sfm_dir.exists():
        return demo_position("no_sfm")

    try:
        model = pycolmap.Reconstruction(str(sfm_dir))

        # Extract features for query
        imgs_dir  = map_dir / "images"
        feats_f   = map_dir / "features.h5"
        pairs_f   = map_dir / "pairs.txt"
        matches_f = map_dir / "matches.h5"

        # Query features
        q_feats = map_dir / "q_features.h5"
        extract_features.main(
            extract_features.confs["superpoint_aachen"],
            query_path.parent,
            image_list=[query_path.name],
            feature_path=q_feats
        )

        # Query pairs
        q_pairs = map_dir / "q_pairs.txt"
        img_list = [f.name for f in sorted(imgs_dir.glob("*.jpg"))][:10]
        pairs_from_exhaustive.main(
            q_pairs,
            image_list=[query_path.name],
            ref_list=img_list
        )

        # Match
        q_matches = map_dir / "q_matches.h5"
        match_features.main(
            match_features.confs["superglue"],
            q_pairs,
            features=feats_f,
            matches=q_matches,
            features_ref=q_feats
        )

        # Localize
        pose = pycolmap.absolute_pose_estimation(
            str(query_path), model,
            pycolmap.Camera(model="SIMPLE_RADIAL", width=640, height=480, params=[700,320,240,0])
        )

        if pose and pose.success:
            pos = pose.cam_from_world.inverse().translation
            return {
                "success":    True,
                "x":          float(pos[0]),
                "y":          float(pos[1]),
                "z":          float(pos[2]),
                "confidence": 0.9,
                "demo":       False
            }
        return demo_position("pose_failed")

    except Exception as e:
        return demo_position(f"hloc_loc: {str(e)[:80]}")

# ── HELPERS ──────────────────────────────────────────────────────────────────
STATUS = {}

def update_status(map_id, status, message=""):
    STATUS[map_id] = {"status": status, "message": message}
    print(f"[{map_id}] {status}: {message}", flush=True)

def demo_position(reason="demo"):
    import random
    return {
        "success":    True,
        "x":          round(random.uniform(0,15), 2),
        "y":          0.0,
        "z":          round(random.uniform(0,25), 2),
        "confidence": 0.3,   # LOW → frontend ignores this
        "demo":       True,
        "reason":     reason
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
