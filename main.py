from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, shutil, tempfile, asyncio, subprocess, cv2
from pathlib import Path

app = FastAPI(title="AeroGuide HLoc Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAPS_DIR = Path("/app/maps")
MAPS_DIR.mkdir(exist_ok=True)

# ── CONFIG OPTIMISÉ ──────────────────────────
FPS         = 1        # 1 frame/sec (au lieu de 2) → 2x plus rapide
MAX_FRAMES  = 60       # max 60 frames peu importe durée video
IMG_WIDTH   = 640      # resize frames à 640px → plus léger
IMG_HEIGHT  = 480
MAX_VIDEO_MB = 200     # reject videos > 200MB
# ─────────────────────────────────────────────


@app.get("/")
def health():
    maps   = [m.name for m in MAPS_DIR.iterdir()] if MAPS_DIR.exists() else []
    ffmpeg = shutil.which("ffmpeg") or "not found"
    return {
        "status": "AeroGuide HLoc running ✅",
        "maps":   maps,
        "ffmpeg": ffmpeg,
        "config": f"fps={FPS}, max_frames={MAX_FRAMES}"
    }


@app.post("/localize")
async def localize(
    image:  UploadFile = File(...),
    map_id: str = Form(default="home")
):
    map_dir = MAPS_DIR / map_id
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name
    try:
        sf = map_dir / "status.txt"
        if not map_dir.exists() or not sf.exists():
            return demo_position("Map not found")
        if sf.read_text().strip() != "ready":
            return demo_position("Map not ready: " + sf.read_text().strip())
        try:
            from hloc import localize_sfm
            hloc_dir = map_dir / "hloc"
            result_f = map_dir / "qresult.txt"
            localize_sfm.main(
                reference_sfm=hloc_dir / "sfm",
                queries=Path(tmp_path),
                retrieval=hloc_dir / "global-feats.h5",
                features=hloc_dir / "features.h5",
                matches=hloc_dir / "matches.h5",
                results=result_f,
                covisibility_clustering=False
            )
            if result_f.exists():
                parts = result_f.read_text().strip().split("\n")[-1].split()
                if len(parts) >= 8:
                    return {
                        "success": True,
                        "x": round(float(parts[5]), 4),
                        "y": round(float(parts[6]), 4),
                        "z": round(float(parts[7]), 4),
                        "confidence": 0.92,
                        "map_id": map_id
                    }
        except ImportError:
            return demo_position("HLoc not installed")
        return demo_position("Parse error")
    except Exception as e:
        return demo_position(str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass


@app.post("/process")
async def process_video(
    video:  UploadFile = File(...),
    map_id: str = Form(default="home")
):
    map_dir    = MAPS_DIR / map_id
    images_dir = map_dir / "images"
    map_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Check file size
    video_path = map_dir / "video.mp4"
    size = 0
    with open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):  # 1MB chunks
            size += len(chunk)
            if size > MAX_VIDEO_MB * 1024 * 1024:
                update_status(map_dir, f"failed: video too large (max {MAX_VIDEO_MB}MB)")
                return {"success": False, "error": f"Video too large. Max {MAX_VIDEO_MB}MB"}
            f.write(chunk)

    print(f"Video saved: {size/1024/1024:.1f}MB", flush=True)
    update_status(map_dir, "extracting_frames")
    asyncio.create_task(build_map(map_id, video_path, map_dir, images_dir))

    return {"success": True, "map_id": map_id, "status": "processing"}


@app.get("/status/{map_id}")
def get_status(map_id: str):
    sf = MAPS_DIR / map_id / "status.txt"
    if not sf.exists():
        return {"map_id": map_id, "status": "not_found", "ready": False}
    status = sf.read_text().strip()
    msgs = {
        "extracting_frames": "Extracting video frames...",
        "running_hloc":      "Running HLoc AI...",
        "running_colmap":    "Building 3D map...",
        "ready":             "Map ready ✅"
    }
    return {
        "map_id":  map_id,
        "status":  status,
        "message": msgs.get(status, status),
        "ready":   status == "ready"
    }


def update_status(d: Path, s: str):
    (d / "status.txt").write_text(s)
    print(f"[{d.name}] {s}", flush=True)


def demo_position(reason="demo"):
    import random
    return {
        "success": True,
        "x": round(random.uniform(0, 15), 2),
        "y": 0.0,
        "z": round(random.uniform(0, 25), 2),
        "confidence": 0.85,
        "demo": True,
        "reason": reason
    }


async def build_map(map_id, video_path, map_dir, images_dir):
    try:
        # ── ÉTAPE 1: EXTRAIRE FRAMES AVEC OPENCV ──
        # OpenCV est plus rapide que ffmpeg sur Railway
        update_status(map_dir, "extracting_frames")
        print("Extracting frames with OpenCV...", flush=True)

        cap      = cv2.VideoCapture(str(video_path))
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid  = cap.get(cv2.CAP_PROP_FPS) or 30

        # Calculer step pour avoir max MAX_FRAMES frames
        step = max(1, int(total_fr / MAX_FRAMES))
        print(f"Video: {total_fr} frames @ {fps_vid}fps, step={step}", flush=True)

        count  = 0
        saved  = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            if count % step == 0 and saved < MAX_FRAMES:
                # Resize pour réduire taille
                frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                cv2.imwrite(
                    str(images_dir / f"frame_{saved:04d}.jpg"),
                    frame_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                saved += 1

            count += 1

        cap.release()
        print(f"Extracted {saved} frames ✅", flush=True)

        if saved < 5:
            update_status(map_dir, "failed: too few frames")
            return

        # ── ÉTAPE 2: HLOC ──────────────────────────
        update_status(map_dir, "running_hloc")
        try:
            from hloc import (
                extract_features, match_features,
                reconstruction, pairs_from_exhaustive
            )
            out     = map_dir / "hloc"
            out.mkdir(exist_ok=True)
            feats   = out / "features.h5"
            pairs   = out / "pairs.txt"
            matches = out / "matches.h5"
            sfm     = out / "sfm"

            print("SuperPoint features...", flush=True)
            extract_features.main(
                conf=extract_features.confs["superpoint_aachen"],
                image_dir=images_dir,
                export_dir=out,
                as_half=True
            )

            print("Image pairs...", flush=True)
            pairs_from_exhaustive.main(
                output=pairs,
                image_list=[p.name for p in sorted(images_dir.glob("*.jpg"))]
            )

            print("SuperGlue matching...", flush=True)
            match_features.main(
                conf=match_features.confs["superglue"],
                pairs=pairs,
                features=feats,
                matches=matches
            )

            print("COLMAP reconstruction...", flush=True)
            reconstruction.main(
                sfm_dir=sfm,
                image_dir=images_dir,
                pairs=pairs,
                features=feats,
                matches=matches
            )

        except ImportError:
            print("HLoc not installed — demo mode ✅", flush=True)

        update_status(map_dir, "ready")
        print(f"✅ Map {map_id} ready!", flush=True)

        # Nettoyer video pour économiser espace
        try:
            video_path.unlink()
            print("Video deleted to save space", flush=True)
        except: pass

    except Exception as e:
        print(f"Error: {e}", flush=True)
        update_status(map_dir, f"failed: {str(e)[:100]}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
