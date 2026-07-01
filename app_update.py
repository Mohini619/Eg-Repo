from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import shutil
import subprocess
import os
import uuid
import time
from PIL import Image

app = FastAPI(title="Pfizer Image Enhancement API")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def format_size(path):
    size=os.path.getsize(path)
    if size<1024:
        return f"{size} B"
    elif size<1024*1024:
        return f"{size/1024:.2f} KB"
    else:
        return f"{size/(1024*1024):.2f} MB"


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Pfizer Image Enhancement</title>
<style>
body{font-family:Arial;background:#f5f5f5;text-align:center;margin:40px;}
.container{background:white;width:520px;margin:auto;padding:25px;border-radius:10px;
box-shadow:0 0 10px rgba(0,0,0,.2);}
table{margin:auto;border-collapse:collapse;}
td,th{padding:8px 14px;border:1px solid #ccc;}
.btn{background:#0066cc;color:white;padding:10px 20px;
border:none;border-radius:5px;text-decoration:none;}
</style>
</head>
<body>
<div class="container">
<h2>Pfizer Image Enhancement</h2>
<form action="/enhance" method="post" enctype="multipart/form-data">
<input type="file" name="file" required><br><br>
<input class="btn" type="submit" value="Enhance Image">
</form>
<br>
<a href="/health">Health Check</a>
</div>
</body>
</html>
"""

@app.get("/health")
def health():
    return {"status":"healthy"}

@app.get("/download/{filename}")
def download(filename:str):
    path=os.path.join(RESULT_DIR,filename)
    if not os.path.exists(path):
        return {"status":"error","message":"File not found"}
    ext=os.path.splitext(filename)[1].lower()
    media="image/png" if ext==".png" else "image/jpeg"
    return FileResponse(path,media_type=media,filename=filename)

@app.post("/enhance",response_class=HTMLResponse)
async def enhance_image(file:UploadFile=File(...)):
    uid=str(uuid.uuid4())
    input_path=os.path.join(UPLOAD_DIR,f"{uid}_{file.filename}")
    with open(input_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    in_img=Image.open(input_path)
    iw,ih=in_img.size

    start=time.time()

    cmd=[
        "python3",
        "inference_realesrgan.py",
        "-n","RealESRGAN_x4plus",
        "--face_enhance",
        "-i",input_path
    ]

    result=subprocess.run(cmd,capture_output=True,text=True)

    if result.returncode!=0:
        return f"<h2>Error</h2><pre>{result.stderr}</pre>"

    name=os.path.splitext(os.path.basename(input_path))[0]

    output_file=None
    for ext in ("png","jpg","jpeg"):
        p=f"{RESULT_DIR}/{name}_out.{ext}"
        if os.path.exists(p):
            output_file=p
            break

    if output_file is None:
        return "<h2>Output file not found.</h2>"

    out_img=Image.open(output_file)
    ow,oh=out_img.size
    elapsed=round(time.time()-start,2)
    factor=round(ow/iw,2)

    download_name=os.path.basename(output_file)

    return f"""
<html>
<head>
<title>Enhancement Report</title>
<style>
body{{font-family:Arial;background:#f5f5f5}}
.container{{width:1100px;margin:25px auto;background:white;padding:25px;border-radius:10px;box-shadow:0 0 10px rgba(0,0,0,.2);text-align:center}}
.tables{{display:flex;justify-content:center;gap:30px}}
table{{border-collapse:collapse;width:450px}}
td,th{{padding:10px;border:1px solid #ccc}}
th{{background:#0b5ed7;color:white}}
.summary{{margin:25px auto;width:500px}}
.btn{{display:inline-block;margin-top:20px;background:#0066cc;color:white;padding:10px 20px;text-decoration:none;border-radius:5px}}
</style>
</head>
<body>
<div class="container">
<h2>Pfizer Image Enhancement Report</h2>
<div class="tables">
<table>
<tr><th colspan="2">Original Image Details</th></tr>
<tr><td>File Name</td><td>{file.filename}</td></tr>
<tr><td>Image Size</td><td>{format_size(input_path)}</td></tr>
<tr><td>Resolution</td><td>{iw} × {ih}</td></tr>
<tr><td>DPI</td><td>{in_img.info.get('dpi',('N/A','N/A'))}</td></tr>
<tr><td>Format</td><td>{in_img.format}</td></tr>
<tr><td>Color Mode</td><td>{in_img.mode}</td></tr>
</table>
<table>
<tr><th colspan="2">Enhanced Image Details</th></tr>
<tr><td>File Name</td><td>{download_name}</td></tr>
<tr><td>Image Size</td><td>{format_size(output_file)}</td></tr>
<tr><td>Resolution</td><td>{ow} × {oh}</td></tr>
<tr><td>DPI</td><td>{out_img.info.get('dpi',('N/A','N/A'))}</td></tr>
<tr><td>Format</td><td>{out_img.format}</td></tr>
<tr><td>Color Mode</td><td>{out_img.mode}</td></tr>
</table>
</div>
<table class="summary">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Enhancement Factor</td><td>{factor}x</td></tr>
<tr><td>Model Used</td><td>RealESRGAN_x4plus + GFPGAN</td></tr>
<tr><td>Processing Time</td><td>{elapsed} seconds</td></tr>
</table>
<a class="btn" href="/download/{download_name}">Download Enhanced Image</a>
</div>
</body>
</html>
"""

