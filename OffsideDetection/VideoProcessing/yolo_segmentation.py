from ultralytics import YOLO
import numpy as np
import os
import cv2
from SoccerNet.Downloader import SoccerNetDownloader

class YOLOSegmentation:
    DEFAULT_MODEL_NAME = "yolo11l-seg.pt"
    DEFAULT_SOCCERNET_DIR = "dataset/SoccerNet"

    def __init__(self, model_path: str = None, device: str = "cpu", conf_threshold: float = 0.3):
        self.model_path = model_path or self.DEFAULT_MODEL_NAME
        self.model = YOLO(self.model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.names = self.model.names
    
    
    
    @staticmethod
    def download_soccernet_data(local_directory: str = None, password: str = None):
        local_directory = local_directory or YOLOSegmentation.DEFAULT_SOCCERNET_DIR
        os.makedirs(local_directory, exist_ok=True)

        downloader = SoccerNetDownloader(LocalDirectory=local_directory)
        if password is not None:
            downloader.password = password
        
        # download labels and tracking annotations
        downloader.downloadGames(
            files=[
                "Labels-v2.json",
                "1_player_boundingbox_maskrcnn.json",
                "2_player_boundingbox_maskrcnn.json",
                "1_field_calib_ccbv.json",
                "2_field_calib_ccbv.json",
            ],
            split=["train", "valid", "test"],
        )

        downloader.downloadGames(
            files=["1_720p.mkv", "2_720p.mkv"],
            split=["train", "valid", "test"],
        )
        return local_directory

    def _get_target_class_ids(self):
        """Return class IDs for players and balls based on the loaded model names."""
        target_ids = {"player": [], "ball": []}
        for class_id, class_name in self.names.items():
            normalized = class_name.lower()
            if normalized in {"player", "person"}:
                target_ids["player"].append(class_id)
            elif normalized in {"ball", "sports ball", "soccer ball"}:
                target_ids["ball"].append(class_id)
        return target_ids

    def detect_players_and_ball(self, frame, conf_threshold: float = None):
        """
        Run inference on a frame and return player and ball detections.

        Returns a dictionary with keys: players, balls, and all_detections.
        """
        conf_threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        results = self.model(frame)
        result = results[0]

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4), dtype=np.float32)
        conf = boxes.conf.cpu().numpy() if len(boxes) else np.zeros((0,), dtype=np.float32)
        cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros((0,), dtype=int)

        target_ids = self._get_target_class_ids()
        players = []
        balls = []
        all_detections = []

        for box, score, class_id in zip(xyxy, conf, cls):
            if score < conf_threshold:
                continue

            class_name = self.names.get(class_id, str(class_id))
            detection = {
                "xyxy": box.tolist(),
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": class_name,
            }
            all_detections.append(detection)

            if class_id in target_ids["player"]:
                players.append(detection)
            elif class_id in target_ids["ball"]:
                balls.append(detection)

        return {
            "players": players,
            "balls": balls,
            "all_detections": all_detections,
        }

    def annotate_frame(self, frame, detections, box_color=(0, 255, 0), text_color=(255, 255, 255)):
        """Draw bounding boxes for player and ball detections."""
        annotated = frame.copy()
        for det in detections.get("all_detections", []):
            x1, y1, x2, y2 = map(int, det["xyxy"])
            class_name = det["class_name"]
            conf = det["confidence"]
            if det in detections.get("players", []):
                color = (0, 255, 0)
            elif det in detections.get("balls", []):
                color = (0, 120, 255)
            else:
                color = box_color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2,
            )
        return annotated

