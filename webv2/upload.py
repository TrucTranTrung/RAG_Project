import uuid
import subprocess
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_ROOT = Path(__file__).parent / "uploads"
IMAGE_DIR = UPLOAD_ROOT / "images"
AUD_DIR = UPLOAD_ROOT / "audio"
VIDEO_DIR = UPLOAD_ROOT / "videos"
ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp"}
ALLOWED_VIDEO_EXTS = {"mp4", "mov", "mkv", "webm"}


def ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    AUD_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def get_duration_seconds(path: Path) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out) if out else None
    except Exception:
        return None


async def save_upload(file: UploadFile, target: Path) -> None:
    ensure_dirs()
    data = await file.read()
    target.write_bytes(data)


def convert_audio_to_mp3(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "128k",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dst.exists():
        raise RuntimeError("Không thể chuyển đổi sang MP3. Kiểm tra cài đặt ffmpeg.")


def build_success(payload: Dict) -> Dict:
    return {"success": True, "payload": payload}


def build_error(message: str) -> Dict:
    return {"success": False, "message": message}


@app.post("/upload")
async def upload(type: str = Form(...), file: UploadFile = File(...)):
    try:
        if not type or not file:
            raise HTTPException(status_code=400, detail="Thiếu dữ liệu tải lên")

        ensure_dirs()

        if type == "audio":
            tmp_name = f"raw_{uuid.uuid4().hex}.webm"
            tmp_path = AUD_DIR / tmp_name
            try:
                await save_upload(file, tmp_path)
            except Exception as e:
                print("Lỗi khi save_upload:", e)
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise

            output_name = f"audio_{uuid.uuid4().hex}.mp3"
            output_path = AUD_DIR / output_name

            try:
                convert_audio_to_mp3(tmp_path, output_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

            duration = get_duration_seconds(output_path)
            return build_success(
                {
                    "type": "audio",
                    "url": f"/uploads/audio/{output_name}",
                    "duration": duration,
                }
            )

        if type == "video":
            ext = get_extension(file.filename or "")
            if ext not in ALLOWED_VIDEO_EXTS:
                raise HTTPException(status_code=400, detail="Định dạng video không được hỗ trợ")

            filename = f"video_{uuid.uuid4().hex}.{ext}"
            target_path = VIDEO_DIR / filename
            await save_upload(file, target_path)

            duration = get_duration_seconds(target_path)
            return build_success(
                {
                    "type": "video",
                    "url": f"/uploads/videos/{filename}",
                    "duration": duration,
                }
            )

        raise HTTPException(status_code=400, detail="Loại tệp không được hỗ trợ")

    except HTTPException as e:
        raise e
    except Exception as e:
        # Trả lỗi dạng giống PHP bản cũ
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "upload:app",
        host="0.0.0.0",
        port=9001,
        reload=True
    )
