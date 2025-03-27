import os
import argparse
import logging
import torch
import cv2
import mediapipe as mp
import numpy as np
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 使用完整範圍模型
            min_detection_confidence=0.5
        )
        
    def detect_and_crop(self, image_path, padding=0.2):
        """
        檢測人臉並裁剪，加入padding來保留更多臉部周圍區域
        padding: 額外擴展的邊界比例
        """
        try:
            # 讀取圖片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("無法讀取圖片")
            
            # 轉換顏色空間
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # 檢測人臉
            results = self.face_detection.process(image_rgb)
            
            if not results.detections:
                logging.warning("未檢測到人臉，將使用原始圖片")
                return Image.fromarray(image_rgb)
            
            # 使用第一個檢測到的人臉（分數最高的）
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # 計算實際像素座標
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # 添加padding
            padding_x = int(w * padding)
            padding_y = int(h * padding)
            
            # 計算擴展後的邊界
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(width, x + w + padding_x)
            y2 = min(height, y + h + padding_y)
            
            # 裁剪圖片
            face_image = image_rgb[y1:y2, x1:x2]
            
            # 轉換為PIL Image
            face_pil = Image.fromarray(face_image)
            
            # 保存裁剪後的圖片用於檢查
            output_path = f"cropped_{os.path.basename(image_path)}"
            face_pil.save(output_path)
            logging.info(f"已保存裁剪後的人臉圖片: {output_path}")
            
            return face_pil
            
        except Exception as e:
            logging.error(f"人臉檢測/裁剪過程中出錯: {str(e)}")
            # 如果出錯，返回原始圖片
            return Image.open(image_path).convert('RGB')

class AvatarScorer:
    def __init__(self, model_path, device='mps' if torch.backends.mps.is_available() else 'cpu'):
        self.device = device
        logging.info(f"Using device: {device}")
        
        # 設置圖片預處理
        self.transform = transforms.Compose([
            transforms.Resize((350, 350)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 初始化人臉檢測器
        self.face_detector = FaceDetector()
        
        # 載入模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path):
        try:
            # 初始化模型架構
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1)
            )
            
            # 檢查模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # 載入模型權重
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            logging.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, image_path):
        try:
            # 檢查圖片文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # 檢測並裁剪人臉
            logging.info("正在檢測和裁剪人臉...")
            face_image = self.face_detector.detect_and_crop(image_path)
            
            # 預處理圖片
            image_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
            
            # 進行預測
            with torch.no_grad():
                prediction = self.model(image_tensor).item()
            
            # 確保預測值在合理範圍內
            prediction = max(1.0, min(5.0, prediction))
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise