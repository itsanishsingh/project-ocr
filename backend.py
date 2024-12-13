from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from main import image_extraction_process, pdf_extraction_process
import aiofiles
import zipfile
from pathlib import Path

app = FastAPI()
output_dir = Path("data/output/")


@app.post("/data_zip")
async def get_data_as_zip(file: UploadFile = File(...)):

    output_dir.mkdir(parents=True, exist_ok=True)

    for file_in_dir in output_dir.iterdir():
        if file_in_dir.is_file():
            file_in_dir.unlink()

    temp_path = f"temp/uploaded_{file.filename}"

    async with aiofiles.open(temp_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    if file.content_type in ["application/pdf"]:
        pdf_extraction_process(temp_path)

    elif file.content_type in ["image/jpg", "image/jpeg", "image/png"]:
        image_extraction_process(temp_path)

    else:
        return {"Error": f"{file.content_type}"}

    zip_path = "output/result.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for img_path in output_dir.glob("*"):
            zipf.write(img_path, Path(img_path).name)

    return FileResponse(zip_path, media_type="application/zip", filename="result.zip")
