from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import shutil

from service.sr_demo import sr_service_demo
from service.inpaint_demo import inpaint_service_demo
from service.deblur_demo import deblur_service_demo


app = FastAPI()

# 创建上传目录（可选，用于保存上传的图片）
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "results"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 原始路由保留
@app.get("/")
def read_root():
    return {"Hello": "Image restoration from zzx"}

# 处理正常图片的演示路由（返回修改过程）
@app.post("/api/demo/sr")
async def demo_sr(file: UploadFile = File(...)):
    """上传正常图片，返回超分辨率处理过程"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        path_to_return = sr_service_demo(
            input_image_path=file_path,
            output_path=OUTPUT_DIR
        )

        return {
            "message": "Image processed successfully",
            "input_path": file_path,
            "output_paths": path_to_return
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
@app.post("/api/demo/deblur")
async def demo_deblur(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        path_to_return = deblur_service_demo(
            input_image_path=file_path,
            output_path=OUTPUT_DIR
        )

        return {
            "message": "Image processed successfully",
            "input_path": file_path,
            "output_paths": path_to_return
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/demo/inpaint")
async def demo_inpaint(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        path_to_return = inpaint_service_demo(
            input_image_path=file_path,
            output_path=OUTPUT_DIR
        )

        return {
            "message": "Image processed successfully",
            "input_path": file_path,
            "output_paths": path_to_return
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# 处理损坏图片的路由（返回处理结果）
@app.post("/api/sr")
async def sr(file: UploadFile = File(...)):
    """上传损坏图片，返回超分辨率处理结果"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # 模拟处理结果（实际可替换为超分辨率算法）
    result = {
        "status": "success",
        "message": f"超分辨率处理完成: {file.filename}",
        "processed_file": f"/processed/{file.filename}"  # 假设的处理后文件路径
    }
    return result

@app.post("/api/deblur")
async def deblur(file: UploadFile = File(...)):
    """上传损坏图片，返回去模糊处理结果"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # 模拟处理结果（实际可替换为去模糊算法）
    result = {
        "status": "success",
        "message": f"去模糊处理完成: {file.filename}",
        "processed_file": f"/processed/{file.filename}"
    }
    return result

@app.post("/api/inpaint")
async def inpaint(file: UploadFile = File(...)):
    """上传损坏图片，返回修复处理结果"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # 模拟处理结果（实际可替换为修复算法）
    result = {
        "status": "success",
        "message": f"修复处理完成: {file.filename}",
        "processed_file": f"/processed/{file.filename}"
    }
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)