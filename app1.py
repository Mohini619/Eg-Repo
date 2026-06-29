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


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pfizer Image Enhancement</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 50px;
                text-align: center;
            }

            .container {
                width: 500px;
                margin: auto;
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }

            h2 {
                color: #0066cc;
            }

            input[type=file] {
                margin: 20px;
            }

            input[type=submit] {
                background-color: #0066cc;
                color: white;
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                border-radius: 5px;
            }

            input[type=submit]:hover {
                background-color: #004c99;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h2>Pfizer Image Enhancement</h2>

            <form action="/enhance" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <br>
                <input type="submit" value="Enhance Image">
            </form>

            <br>

            <a href="/health">Health Check</a>
        </div>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/enhance", response_class=HTMLResponse)
async def enhance_image(file: UploadFile = File(...)):

    uid = str(uuid.uuid4())

    input_path = os.path.join(
        UPLOAD_DIR,
        f"{uid}_{file.filename}"
    )

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    original_img = Image.open(input_path)
    original_width, original_height = original_img.size

    start_time = time.time()

    cmd = [
        "python3",
        "inference_realesrgan.py",
        "-n",
        "RealESRGAN_x4plus",
        "--face_enhance",
        "-i",
        input_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return f"""
        <h2>Enhancement Failed</h2>
        <pre>{result.stderr}</pre>
        """

    filename = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    png_file = f"results/{filename}_out.png"
    jpg_file = f"results/{filename}_out.jpg"
    jpeg_file = f"results/{filename}_out.jpeg"

    output_file = None

    if os.path.exists(png_file):
        output_file = png_file

    elif os.path.exists(jpg_file):
        output_file = jpg_file

    elif os.path.exists(jpeg_file):
        output_file = jpeg_file

    else:
        return f"""
        <h2>Error</h2>
        Output file not found.
        """

    end_time = time.time()

    enhanced_img = Image.open(output_file)
    enhanced_width, enhanced_height = enhanced_img.size

    processing_time = round(
        end_time - start_time,
        2
    )

    enhancement_factor = round(
        enhanced_width / original_width,
        2
    )

    download_filename = os.path.basename(output_file)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhancement Report</title>

        <style>
            body {{
                font-family: Arial;
                text-align: center;
                margin: 40px;
            }}

            .container {{
                width: 750px;
                margin: auto;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}

            td, th {{
                border: 1px solid #ddd;
                padding: 12px;
            }}

            th {{
                background-color: #0066cc;
                color: white;
            }}

            h2 {{
                color: #0066cc;
            }}

            .btn {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 20px;
                background: #0066cc;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}

            .btn:hover {{
                background: #004c99;
            }}
        </style>

    </head>

    <body>

        <div class="container">

            <h2>Pfizer Image Enhancement Report</h2>

            <table>

                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>

                <tr>
                    <td>Input File</td>
                    <td>{file.filename}</td>
                </tr>

                <tr>
                    <td>Original Resolution</td>
                    <td>{original_width} x {original_height}</td>
                </tr>

                <tr>
                    <td>Enhanced Resolution</td>
                    <td>{enhanced_width} x {enhanced_height}</td>
                </tr>

                <tr>
                    <td>Enhancement Factor</td>
                    <td>{enhancement_factor}x</td>
                </tr>

                <tr>
                    <td>Model Used</td>
                    <td>GFPGAN + RealESRGAN_x4plus</td>
                </tr>

                <tr>
                    <td>Processing Time</td>
                    <td>{processing_time} Seconds</td>
                </tr>

            </table>

            <a class="btn" href="/download/{download_filename}">
                Download Enhanced Image
            </a>

            <br><br>

            <a href="/">
                Enhance Another Image
            </a>

        </div>

    </body>
    </html>
    """


@app.get("/download/{filename}")
def download_file(filename: str):

    file_path = os.path.join(
        RESULT_DIR,
        filename
    )

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )
