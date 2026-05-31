from ultralytics import YOLO
import numpy as np
import os
import cv2
import json
from pathlib import Path
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
        
        print(f"Downloading SoccerNet bounding box annotations...")
        print(f"  This downloads annotation files (~3GB, no video files)")
        
        # Download bounding box annotations (no videos, to keep size small)
        try:
            downloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
            print(f"✓ Downloaded tracking annotations")
        except Exception as e:
            print(f"Note: Could not download tracking task: {e}")
        
        try:
            downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
            print(f"✓ Downloaded tracking-2023 annotations")
        except Exception as e:
            print(f"Note: Could not download tracking-2023 task: {e}")

        print(f"✓ SoccerNet data downloaded to {local_directory}")
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

    @staticmethod
    def convert_soccernet_to_yolo(soccernet_dir: str = None, output_dir: str = "dataset/player_detection",
                                  frame_width: int = 1280, frame_height: int = 720):
        """
        Convert SoccerNet bounding box JSON files to YOLO txt format.
        Extracts player bounding boxes and creates label files in YOLO format.
        
        YOLO format: class_id x_center y_center width height (all normalized 0-1)
        """
        soccernet_dir = Path(soccernet_dir or YOLOSegmentation.DEFAULT_SOCCERNET_DIR)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        total_boxes = 0
        
        # Find all boundingbox_maskrcnn.json files
        json_files = list(soccernet_dir.rglob("boundingbox_maskrcnn.json"))
        print(f"Found {len(json_files)} annotation files in {soccernet_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Get the match directory
                match_dir = json_file.parent
                match_name = json_file.parent.name
                print(f"Processing {match_name}...")
                
                # Process each frame annotation
                frame_num = 0
                for frame_data in data:
                    if 'players' not in frame_data or not frame_data['players']:
                        frame_num += 1
                        continue
                    
                    # Create label file
                    label_filename = labels_dir / f"{match_name}_frame_{frame_num}.txt"
                    
                    with open(label_filename, 'w') as label_file:
                        frame_has_boxes = False
                        for player in frame_data['players']:
                            # Player format: [x1, y1, x2, y2] in pixel coordinates
                            boundingbox = player.get('boundingbox', [])
                            if not boundingbox or len(boundingbox) != 4:
                                continue
                            
                            x1, y1, x2, y2 = boundingbox
                            
                            # Clamp to frame boundaries
                            x1 = max(0, min(x1, frame_width))
                            x2 = max(0, min(x2, frame_width))
                            y1 = max(0, min(y1, frame_height))
                            y2 = max(0, min(y2, frame_height))
                            
                            # Skip invalid boxes
                            if x1 >= x2 or y1 >= y2:
                                continue
                            
                            # Convert to YOLO format: normalized center coordinates
                            x_center = (x1 + x2) / 2.0 / frame_width
                            y_center = (y1 + y2) / 2.0 / frame_height
                            width = (x2 - x1) / frame_width
                            height = (y2 - y1) / frame_height
                            
                            # Class 0 = player
                            label_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            total_boxes += 1
                            frame_has_boxes = True
                    
                    if frame_has_boxes:
                        converted_count += 1
                    frame_num += 1
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        print(f"✓ Converted {converted_count} frames with {total_boxes} player boxes to {output_dir}")
        return str(output_dir), converted_count, total_boxes

    @staticmethod
    def create_minimal_test_data(output_dir: str = "dataset/yolo_player_dataset", num_samples: int = 50):
        """
        Create minimal test dataset for debugging/quick testing.
        Generates synthetic images with random player annotations.
        """
        output_dataset = Path(output_dir)
        print(f"Creating minimal test dataset with {num_samples} samples...")
        
        for split in ["train", "val", "test"]:
            (output_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Split data
        train_count = int(0.7 * num_samples)
        val_count = int(0.15 * num_samples)
        
        sample_idx = 0
        for split, count in [("train", train_count), ("val", val_count), ("test", num_samples - train_count - val_count)]:
            for i in range(count):
                # Create synthetic image
                img = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
                img_path = output_dataset / "images" / split / f"sample_{sample_idx}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # Create label with random player box
                label_path = output_dataset / "labels" / split / f"sample_{sample_idx}.txt"
                with open(label_path, 'w') as f:
                    # Random center coordinates and size
                    x_center = np.random.uniform(0.1, 0.9)
                    y_center = np.random.uniform(0.1, 0.9)
                    width = np.random.uniform(0.05, 0.3)
                    height = np.random.uniform(0.1, 0.4)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                sample_idx += 1
        
        print(f"✓ Created minimal test dataset at {output_dataset}")
        return str(output_dataset)
        """
        Generate synthetic placeholder images for labels (when actual video frames not available).
        Creates random images that match YOLO label format for training.
        """
        labels_path = Path(labels_dir)
        images_path = Path(images_dir)
        images_path.mkdir(parents=True, exist_ok=True)
        
        # Get all label files
        label_files = list(labels_path.glob("*.txt"))
        print(f"  Found {len(label_files)} label files in {labels_dir}")
        
        generated_count = 0
        for label_file in label_files:
            image_name = label_file.stem + '.jpg'
            image_file = images_path / image_name
            
            # Generate a synthetic image (random noise as placeholder)
            synthetic_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            cv2.imwrite(str(image_file), synthetic_img)
            generated_count += 1
            
            if generated_count % 100 == 0:
                print(f"    Generated {generated_count} images...")
        
        print(f"✓ Generated {generated_count} synthetic images in {images_dir}")
        return generated_count

    @staticmethod
    def create_training_dataset(converted_data_dir: str, output_dataset_dir: str = "dataset/yolo_player_dataset", 
                               generate_images: bool = True):
        """
        Create YOLO dataset structure with train/val/test splits.
        Organizes converted label files into proper YOLO directory structure.
        Optionally generates synthetic images if actual frames are unavailable.
        """
        converted_dir = Path(converted_data_dir)
        output_dataset = Path(output_dataset_dir)
        
        print(f"Creating dataset structure at {output_dataset}...")
        for split in ["train", "val", "test"]:
            (output_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Get all label files from converted directory
        all_labels = sorted(list((Path(converted_data_dir) / "labels").glob("*.txt")))
        print(f"Found {len(all_labels)} total label files")
        
        if len(all_labels) == 0:
            print("WARNING: No label files found! Check if SoccerNet data was downloaded correctly.")
            return str(output_dataset)
        
        # Split into train/val/test (70% train, 15% val, 15% test)
        total = len(all_labels)
        train_count = int(0.7 * total)
        val_count = int(0.15 * total)
        test_count = total - train_count - val_count
        
        print(f"Splitting data: {train_count} train, {val_count} val, {test_count} test")
        
        split_assignments = {}
        for idx, label_file in enumerate(all_labels):
            if idx < train_count:
                split = "train"
            elif idx < train_count + val_count:
                split = "val"
            else:
                split = "test"
            
            split_assignments[label_file.name] = split
            
            # Copy label file
            dest_label = output_dataset / "labels" / split / label_file.name
            with open(label_file, 'r') as f:
                content = f.read()
            with open(dest_label, 'w') as f:
                f.write(content)
        
        # Generate synthetic images if requested
        if generate_images:
            print("Generating synthetic images...")
            for split in ["train", "val", "test"]:
                labels_split_dir = output_dataset / "labels" / split
                images_split_dir = output_dataset / "images" / split
                print(f"\n[{split.upper()}]")
                YOLOSegmentation.generate_synthetic_images(
                    labels_dir=str(labels_split_dir),
                    images_dir=str(images_split_dir)
                )
        
        print(f"\n✓ Dataset structure created at {output_dataset}")
        return str(output_dataset)

    @staticmethod
    def create_data_yaml(dataset_dir: str, output_yaml: str = "data_player.yaml"):
        """Create data.yaml configuration for YOLO training."""
        dataset_path = Path(dataset_dir).resolve()
        yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['player']
"""
        
        output_path = Path(output_yaml)
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created data.yaml at {output_path}")
        return str(output_path)

    @staticmethod
    def train_on_soccernet(soccernet_dir: str = None, epochs: int = 50, batch_size: int = 16, 
                          device: str = "0", img_size: int = 640, use_test_data: bool = False):
        """
        Train a custom YOLO model on SoccerNet player detection data.
        
        Args:
            soccernet_dir: Path to SoccerNet dataset directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: CUDA device ID (e.g., "0") or "cpu"
            img_size: Input image size for YOLO
            use_test_data: If True, create minimal test data instead of using SoccerNet
        
        Returns:
            Trained model path
        """
        print("\n" + "="*70)
        print("YOLO Player Detection Training")
        print("="*70)
        
        # Step 0: Check if using test data or SoccerNet data
        if use_test_data:
            print("\n[0/3] Creating minimal test dataset for debugging...")
            dataset_dir = YOLOSegmentation.create_minimal_test_data(num_samples=100)
        else:
            # Step 1: Convert SoccerNet annotations to YOLO format
            print("\n[1/4] Converting SoccerNet annotations to YOLO format...")
            converted_dir, frame_count, box_count = YOLOSegmentation.convert_soccernet_to_yolo(
                soccernet_dir=soccernet_dir,
                output_dir="dataset/player_detection"
            )
            
            if frame_count == 0:
                print("\nERROR: No frames were converted!")
                print("This likely means SoccerNet data wasn't downloaded properly.")
                print("\nTroubleshooting:")
                print("  1. Check that 'dataset/SoccerNet' directory exists")
                print("  2. Check that it contains subdirectories like 'train', 'test', 'challenge'")
                print("  3. Check that boundingbox_maskrcnn.json files exist in those directories")
                print("\nTo use minimal test data instead: python yolo_segmentation.py test")
                return None
            
            print(f"     ✓ Converted {frame_count} frames with {box_count} player boxes")
            
            # Step 2: Create training dataset structure
            print("\n[2/4] Creating training dataset structure with synthetic images...")
            dataset_dir = YOLOSegmentation.create_training_dataset(
                converted_data_dir=converted_dir,
                output_dataset_dir="dataset/yolo_player_dataset",
                generate_images=True
            )
        
        # Step 3: Create data.yaml
        print(f"\n[3/4] Creating data.yaml configuration...")
        data_yaml = YOLOSegmentation.create_data_yaml(
            dataset_dir=dataset_dir,
            output_yaml="data_player.yaml"
        )
        
        # Step 4: Train YOLO model
        print("\n[4/4] Training YOLO model...")
        print(f"     Epochs: {epochs}, Batch: {batch_size}, Device: {device}, Image size: {img_size}")
        print(f"     Using detection model (yolo11l.pt) for bounding box data")
        print(f"     Note: SoccerNet provides bounding boxes only, not segmentation masks")
        
        # Use detection model (yolo11l.pt) instead of segmentation (yolo11l-seg.pt)
        # because SoccerNet only provides bounding boxes, not segmentation masks
        model = YOLO("yolo11l.pt")
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            patience=5,
            save=True,
            project="runs/detect",
            name="player_detection",
            pretrained=True,
            optimizer='SGD'
        )
        
        print("\n✓ Training completed!")
        print(f"Results saved to: {results.save_dir}")
        
        # Return path to best model
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        print(f"✓ Best model saved to: {best_model_path}")
        print("="*70)
        
        return str(best_model_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode with SoccerNet data
        print("YOLO Training with SoccerNet Data")
        print("-" * 60)
        
        # Download SoccerNet data if password provided
        if len(sys.argv) > 2:
            password = sys.argv[2]
            print(f"\nDownloading SoccerNet data (password provided)...")
            YOLOSegmentation.download_soccernet_data(password=password)
        else:
            print("\nUsing existing SoccerNet data...")
            print("If you haven't downloaded it yet, run: python yolo_segmentation.py download <password>")
        
        # Train model
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
        device = sys.argv[5] if len(sys.argv) > 5 else "0"
        
        best_model = YOLOSegmentation.train_on_soccernet(
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            use_test_data=False
        )
        if best_model:
            print(f"\n✓ Training complete! Best model: {best_model}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test with minimal synthetic data
        print("YOLO Training with Minimal Test Data")
        print("-" * 60)
        print("This uses synthetic data for quick testing/debugging.\n")
        
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        device = sys.argv[4] if len(sys.argv) > 4 else "0"
        
        best_model = YOLOSegmentation.train_on_soccernet(
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            use_test_data=True
        )
        if best_model:
            print(f"\n✓ Test training complete! Model: {best_model}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "download":
        # Download only
        password = sys.argv[2] if len(sys.argv) > 2 else None
        print("Downloading SoccerNet Data")
        print("-" * 60)
        if not password:
            print("ERROR: Password required for download")
            print("Usage: python yolo_segmentation.py download <password>")
        else:
            YOLOSegmentation.download_soccernet_data(password=password)
            print("✓ Download complete!")
        
    else:
        # Help
        print("="*70)
        print("YOLO Player Detection Training Pipeline")
        print("="*70)
        print("\nUsage:")
        print("  Download SoccerNet data:")
        print("    python yolo_segmentation.py download <password>")
        print("\n  Train on SoccerNet data:")
        print("    python yolo_segmentation.py train [password] [epochs] [batch_size] [device]")
        print("    python yolo_segmentation.py train mypassword 50 16 0")
        print("\n  Quick test with synthetic data:")
        print("    python yolo_segmentation.py test [epochs] [batch_size] [device]")
        print("    python yolo_segmentation.py test 5 8 0")
        print("\nExample workflow:")
        print("  1. python yolo_segmentation.py download <your-password>")
        print("  2. python yolo_segmentation.py train <your-password> 50 16 0")
        print("\nFor quick testing (no SoccerNet data needed):")
        print("  python yolo_segmentation.py test 5 8 0")
        print("="*70)
