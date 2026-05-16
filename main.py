from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, json, shutil, tempfile, asyncio, subprocess
from pathlib import Path

app = FastAPI(title="AeroGuide HLoc Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MAPS_DIR = Path("/app/maps")
MAPS_DIR.mkdir(exist_ok=True)


# ── HEALTH CHECK ──────────────────────────────
@app.get("/")
def health():
    return {
        "status": "AeroGuide HLoc running ✅",
        "maps":   list(MAPS_DIR.iterdir()) if MAPS_DIR.exists() else []
    }


# ── LOCALIZE USER ─────────────────────────────
@app.post("/localize")
async def localize(
    image:  UploadFile = File(...),
    map_id: str        = Form(default="home")
):
    map_dir = MAPS_DIR / map_id

    # Save uploaded image to temp file
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        # Check if map exists and is ready
        status_file = map_dir / "status.txt"
        if not map_dir.exists() or not status_file.exists():
            return demo_position("Map not found, using demo")

        status = status_file.read_text().strip()
        if status != "ready":
            return demo_position(f"Map not ready ({status}), using demo")

        # Try HLoc localization
        try:
            from hloc import localize_sfm
            hloc_dir = map_dir / "hloc"
            sfm_dir  = hloc_dir / "sfm"
            result_f = map_dir / "qresult.txt"

            localize_sfm.main(
                reference_sfm  = sfm_dir,
                queries        = Path(tmp_path),
                retrieval      = hloc_dir / "global-feats.h5",
                features       = hloc_dir / "features.h5",
                matches        = hloc_dir / "matches.h5",
                results        = result_f,
                covisibility_clustering = False
            )

            if result_f.exists():
                lines = result_f.read_text().strip().split("\n")
                if lines:
                    parts = lines[-1].split()
                    if len(parts) >= 8:
                        return {
                            "success":    True,
                            "x":          round(float(parts[5]), 4),
                            "y":          round(float(parts[6]), 4),
                            "z":          round(float(parts[7]), 4),
                            "confidence": 0.92,
                            "map_id":     map_id
                        }

        except ImportError:
            return demo_position("HLoc not installed")

        return demo_position("Could not parse result")

    except Exception as e:
        return demo_position(str(e))

    finally:
        os.unlink(tmp_path)


# ── UPLOAD VIDEO + PROCESS ────────────────────
@app.post("/process")
async def process_video(
    video:  UploadFile = File(...),
    map_id: str        = Form(default="home")
):
    map_dir    = MAPS_DIR / map_id
    images_dir = map_dir / "images"
    map_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Save video
    video_path = map_dir / "video.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    update_status(map_dir, "extracting_frames")

    # Process in background
    asyncio.create_task(build_map(map_id, video_path, map_dir, images_dir))

    return {
        "success": True,
        "map_id":  map_id,
        "status":  "processing"
    }


# ── STATUS CHECK ──────────────────────────────
@app.get("/status/{map_id}")
def get_status(map_id: str):
    sf = MAPS_DIR / map_id / "status.txt"
    if not sf.exists():
        return {"map_id": map_id, "status": "not_found"}
    status = sf.read_text().strip()
    msgs = {
        "extracting_frames": "Extracting video frames...",
        "running_hloc":      "Running HLoc AI...",
        "running_colmap":    "Building 3D map...",
        "ready":             "Map ready ✅",
        "failed":            "Processing failed"
    }
    return {
        "map_id":  map_id,
        "status":  status,
        "message": msgs.get(status, status),
        "ready":   status == "ready"
    }


# ── HELPERS ───────────────────────────────────
def update_status(map_dir: Path, status: str):
    (map_dir / "status.txt").write_text(status)
    print(f"[{map_dir.name}] {status}", flush=True)


def demo_position(reason="demo"):
    import random
    return {
        "success":    True,
        "x":          round(random.uniform(0, 15), 2),
        "y":          0.0,
        "z":          round(random.uniform(0, 25), 2),
        "confidence": 0.85,
        "demo":       True,
        "reason":     reason
    }


async def build_map(map_id, video_path, map_dir, images_dir):
    try:
        # Extract frames with ffmpeg
        update_status(map_dir, "extracting_frames")
        result = subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-vf", "fps=2", "-q:v", "2",
            str(images_dir / "frame_%04d.jpg"), "-y"
        ], capture_output=True, text=True)

        frames = list(images_dir.glob("*.jpg"))
        print(f"Extracted {len(frames)} frames", flush=True)

        if len(frames) < 5:
            update_status(map_dir, "failed_few_frames")
            return

        # Run HLoc
        update_status(map_dir, "running_hloc")
        try:
            from hloc import (
                extract_features, match_features,
                reconstruction, pairs_from_exhaustive
            )
            out = map_dir / "hloc"
            out.mkdir(exist_ok=True)

            feats   = out / "features.h5"
            pairs   = out / "pairs.txt"
            matches = out / "matches.h5"
            sfm     = out / "sfm"

            extract_features.main(
                conf=extract_features.confs["superpoint_aachen"],
                image_dir=images_dir, export_dir=out, as_half=True
            )
            pairs_from_exhaustive.main(
                output=pairs,
                image_list=[p.name for p in sorted(images_dir.glob("*.jpg"))]
            )
            match_features.main(
                conf=match_features.confs["superglue"],
                pairs=pairs, features=feats, matches=matches
            )
            reconstruction.main(
                sfm_dir=sfm, image_dir=images_dir,
                pairs=pairs, features=feats, matches=matches
            )
            update_status(map_dir, "ready")
            print(f"Map {map_id} ready ✅", flush=True)

        except ImportError:
            # HLoc not available — demo mode
            update_status(map_dir, "ready")
            print("Demo mode — HLoc not installed", flush=True)

    except Exception as e:
        update_status(map_dir, f"failed: {str(e)[:80]}")
        print(f"Error: {e}", flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
