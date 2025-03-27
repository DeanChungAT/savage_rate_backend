import os
import sys
import logging
import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile

# 添加項目根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.beauty_model import AvatarScorer
from comment_template import COMMENTS

# 設置日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型
try:
    logger.info("開始初始化模型")
    model_path = os.path.join("models", "best_model.pth")
    scorer = AvatarScorer(model_path)
    logger.info("模型初始化成功")
except Exception as e:
    logger.error(f"模型初始化錯誤: {str(e)}")
    raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("接收到新的預測請求")
        
        # 創建臨時文件保存上傳的圖片
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # 進行預測
        logger.info("開始進行預測")
        score = scorer.predict(temp_path)
        
        # 刪除臨時文件
        os.unlink(temp_path)
        
        # 四捨五入到一位小數
        rounded_score = round(float(score), 1)
        # 確保分數在 1.0 到 5.0 之間
        rounded_score = max(1.0, min(5.0, rounded_score))
        
        # 獲取評論
        comments = COMMENTS.get(rounded_score, ["你的顏值無法用語言形容"])
        comment = random.choice(comments)
        
        logger.info(f"預測完成，分數: {rounded_score}, 評論: {comment}")
        return {"score": rounded_score, "comment": comment}
        
    except Exception as e:
        logger.error(f"預測過程發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 