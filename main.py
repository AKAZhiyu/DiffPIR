from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import os
import shutil

from service.sr_demo import sr_service_demo
from service.inpaint_demo import inpaint_service_demo
from service.deblur_demo import deblur_service_demo

from service.sr import sr_service
from service.inpaint import inpaint_service
from service.deblur import deblur_service


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


@app.post("/api/inpaint")
async def inpaint(file: UploadFile = File(...), mask_file: UploadFile = File(...)):
    """上传图片和mask文件，进行图像修复"""
    try:
        # 保存上传的图片和mask文件
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        mask_path = os.path.join(UPLOAD_DIR, mask_file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with open(mask_path, "wb") as buffer:
            shutil.copyfileobj(mask_file.file, buffer)

        # 调用 inpaint_service 函数处理图像
        path_to_return = inpaint_service(
            mask_type='box',  # 默认使用 'box'，可根据需求调整
            input_image_path=file_path,
            mask_path=mask_path,
            output_path=OUTPUT_DIR
        )

        # 返回处理结果
        return {
            "message": "Image processed successfully",
            "input_path": file_path,
            "mask_path": mask_path,
            "output_paths": path_to_return
        }

    except Exception as e:
        # 处理异常并返回错误信息
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(mask_path):
            os.remove(mask_path)

@app.post("/api/sr")
async def sr(
    file: UploadFile = File(...),
    scale_factor: int = Query(4, description="Scale factor for super-resolution")
):
    """上传低分辨率图片，进行超分辨率处理"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        path_to_return = sr_service(
            input_image_path=file_path,
            output_path=OUTPUT_DIR,
            is_low_resolution=True,
            scale_factor=scale_factor
        )

        return {
            "message": "Image processed successfully",
            "input_path": file_path,
            "scale_factor": scale_factor,
            "output_paths": path_to_return
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/deblur")
async def deblur(
    file: UploadFile = File(...),
    custom_kernel_size: int = Query(21, description="Kernel size for deblurring"),
    custom_kernel_std: float = Query(1.5, description="Kernel standard deviation for deblurring")
):
    """上传模糊图片，进行去模糊处理"""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        path_to_return = deblur_service(
            blur_mode='Gaussian',
            input_image_path=file_path,
            output_path=OUTPUT_DIR,
            is_blurred=True,
            custom_kernel_size=custom_kernel_size,
            custom_kernel_std=custom_kernel_std
        )

        return {
            "message": "Image processed successfully",
            "input_path": file_path,
            "custom_kernel_size": custom_kernel_size,
            "custom_kernel_std": custom_kernel_std,
            "output_paths": path_to_return
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
