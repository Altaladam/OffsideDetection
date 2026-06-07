from anyio import sleep
import cv2, pafy, os
from cv2.gapi import video
from pytube import YouTube
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from VideoProcessing.yolo_segmentation import YOLOSegmentation
from VideoProcessing.functions import get_average_color, classify_bgr_color
import supervision as sv
from inference.models.utils import get_roboflow_model
import json


class VideoProcessing():

    def read():
    #     url   = "https://www.youtube.com/watch?v=3N7BkyuEBAw&ab_channel=HashtagUnited&t=6090s"
    #     def is_video_downloaded(url):
    #         return os.path.exists(f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4")
    #     if not is_video_downloaded(url):
    #         video = yt_dlp.YoutubeDL().download(url)
    #     else:
    #         video = f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4"

        
        video = f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4"
        capture = cv2.VideoCapture(video)
        
        backSub = cv2.createBackgroundSubtractorKNN() 
        
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while(capture.isOpened()):
            check, frame = capture.read()
            if check == True:
                fgMask = backSub.apply(frame)
                
                kernel = np.ones((4,4),np.uint8)
                opening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
                
                #opening = cv2.medianBlur(opening, 5)

                colored = cv2.bitwise_and(frame, frame, mask = opening);
                frame = cv2.resize(frame, (1080,720))
                colored = cv2.resize(colored, (1080,720))
                cv2.imshow('frame',frame)
                cv2.imshow('FG Mask', colored)
                cv2.waitKey(30)
                            
            else:
                break

        capture.release()
        cv2.destroyAllWindows()
    
    
    def YOLO():
        video = f"hashtag_united_short.mp4"
        #video = f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4"
        #video = f"vid.mov"
        
        cap = cv2.VideoCapture(video)
        
        # # Set video start time to 17:54
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # start_time_seconds = 17 * 60 + 54
        # start_frame = int(start_time_seconds * fps)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        MODEL_NAME = "yolo11l-seg.pt"
        yolo_detector = YOLOSegmentation(device="cuda", conf_threshold=0.3)
        #YOLOSegmentation.download_soccernet_data(local_directory="dataset/SoccerNet", password="s0cc3rn3t")
        model = YOLO(MODEL_NAME)
        model.to('cuda')
        CONF_THRESHOLD = 0.3

        # Get the first frame for pitch corner calibration
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video.")
            return

        # Try to load saved calibration, otherwise manual calibration
        calibration_file = "pitch_calibration.json"
        corners, use_enhanced = VideoProcessing.load_or_calibrate_corners(frame, calibration_file)
        if corners is None or len(corners) < 4:
            print("Calibration failed.")
            return
        print(f"Using pitch calibration: {len(corners)} points ({('Enhanced' if use_enhanced else 'Basic')} mode)")
        
        if use_enhanced and len(corners) == 8:
            # Enhanced calibration with 8 points for better accuracy
            pitch_points = VideoProcessing.get_enhanced_pitch_points()
        else:
            # Basic 4-corner calibration - horizontal pitch (width=800, height=500)
            pitch_points = [(0, 0), (800, 0), (800, 500), (0, 500)]
        
        # Define known team colors in HSV
        
        #team1_hsv = np.array([251, 157, 221])  # colors for vid.mov
        #team2_hsv = np.array([13, 28, 103])
        team1_hsv = np.array([45, 120, 170])    # colors for FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4
        team2_hsv = np.array([90, 100, 140])
        referee_hsv = np.array([0, 30, 60])     # Black/dark colors for referee

        # Initialize ByteTrack tracker for smooth player trajectories
        byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=90,             # Keep track alive for 90 frames (3 seconds at 30fps) when lost
            minimum_matching_threshold=0.7,
            frame_rate=30                     # Assumed frame rate
        )
        
        # Store team assignments per track_id for consistency
        track_team_assignments = {}  # {track_id: team_label}
        
        # Camera motion estimation variables
        prev_gray = None
        camera_motion_transform = np.eye(3, dtype=np.float32)  # Identity matrix initially
        
        # Feature detector for camera motion estimation (using ORB for speed)
        feature_detector = cv2.ORB_create(nfeatures=500)
        
        # Feature matcher
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


        p = 0

        while True:
            
            # For testing and comparison - pause after 1000 frames
            # p += 1
            # if p == 330:
            #     cv2.waitKey(0)
            
            if not ret:
                break
            
            # Camera motion estimation
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Detect features in both frames
                kp1, des1 = feature_detector.detectAndCompute(prev_gray, None)
                kp2, des2 = feature_detector.detectAndCompute(curr_gray, None)
                
                if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                    # Match features
                    matches = bf_matcher.match(des1, des2)
                    
                    if len(matches) > 10:
                        # Sort matches by distance
                        matches = sorted(matches, key=lambda x: x.distance)
                        
                        # Use top 50% of matches
                        good_matches = matches[:len(matches)//2]
                        
                        if len(good_matches) >= 4:
                            # Extract matched keypoints
                            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            # Estimate affine transformation (camera motion)
                            transform_matrix, inliers = cv2.estimateAffinePartial2D(
                                src_pts, dst_pts, 
                                method=cv2.RANSAC, 
                                ransacReprojThreshold=3.0
                            )
                            
                            if transform_matrix is not None:
                                # Convert 2x3 affine to 3x3 homogeneous
                                camera_motion_transform = np.vstack([transform_matrix, [0, 0, 1]])
                            else:
                                camera_motion_transform = np.eye(3, dtype=np.float32)
                        else:
                            camera_motion_transform = np.eye(3, dtype=np.float32)
                    else:
                        camera_motion_transform = np.eye(3, dtype=np.float32)
                else:
                    camera_motion_transform = np.eye(3, dtype=np.float32)
            
            prev_gray = curr_gray.copy()

            #Generic YOLOv11 detection code
            
            # results = model(frame)
            # boxes = results[0].boxes
            # mask = boxes.conf.cpu().numpy() > CONF_THRESHOLD
            # filtered_boxes = boxes[mask]
            
            # # Convert YOLO detections to supervision Detections format for ByteTrack
            # if len(filtered_boxes) > 0:
            #     xyxy = filtered_boxes.xyxy.cpu().numpy()
            #     conf = filtered_boxes.conf.cpu().numpy()
            #     cls = filtered_boxes.cls.cpu().numpy() if filtered_boxes.cls is not None else np.zeros(len(conf))
                
            #     detections = sv.Detections(
            #         xyxy=xyxy,
            #         confidence=conf,
            #         class_id=cls.astype(int)
            #     )

            #Custom trained YOLOv11 segmentation model

            detections = yolo_detector.detect_players_and_ball(frame)
            players = detections["players"]  # Only use this
            
            # Convert YOLO detections to supervision Detections format for ByteTrack
            if len(players) > 0:
                xyxy = np.array([d['xyxy'] for d in players], dtype=np.float32)
                conf = np.array([d['confidence'] for d in players], dtype=np.float32)
                cls = [d['class_id'] if d['class_id'] is not None else 0 for d in players]
                
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=np.array(cls).astype(int)
                )
                # Comment until here to use generic YOLOv11 detections instead of custom model
                
                # Apply camera motion compensation to improve tracking during camera movement
                # Transform detection boxes to account for camera motion between frames
                if camera_motion_transform is not None and not np.allclose(camera_motion_transform, np.eye(3), atol=0.01):
                    # Get the translation and scale from the transform
                    tx = camera_motion_transform[0, 2]  # X translation
                    ty = camera_motion_transform[1, 2]  # Y translation
                    
                    # Log significant camera motion
                    if abs(tx) > 5 or abs(ty) > 5:
                        print(f"Camera motion detected: dx={tx:.1f}, dy={ty:.1f}")
                
                # Update tracker with detections
                tracked_detections = byte_tracker.update_with_detections(detections)
            else:
                tracked_detections = sv.Detections.empty()

            colors = []
            avg_hsvs = []
            player_centers = []
            track_ids = []
            
            for i in range(len(tracked_detections)):
                x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
                track_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else i
                
                h, w = y2 - y1, x2 - x1
                if h <= 0 or w <= 0:
                    continue
                cy1 = y1 + int(0.2 * h)
                cy2 = y2 - int(0.2 * h)
                cx1 = x1 + int(0.2 * w)
                cx2 = x2 - int(0.2 * w)
                roi = frame[cy1:cy2, cx1:cx2]
                if roi.size == 0:
                    continue
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                avg_hue = np.mean(hsv_roi[:, :, 0])
                colors.append([avg_hue])
                avg_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)
                avg_hsvs.append(avg_hsv)
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                player_centers.append(center)
                track_ids.append(track_id)

            offside_indices = set()
            mapped_points = []
            final_labels = []
            if len(colors) >= 2:
                Z = np.float32(colors)
                K = 3  # number of clusters (2 teams + referee)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret_km, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                cluster_colors = [(0, 255, 255), (255, 0, 255), (128, 128, 128)]  # cyan, magenta, gray for referee

                # Combine k-means and color-based assignment with referee detection
                final_labels = VideoProcessing.color_based_team_assignment_with_referee(
                    avg_hsvs, label.flatten(), team1_hsv, team2_hsv, referee_hsv, track_ids, threshold=15
                )

                # Homography: Map player centers to 2D pitch
                if len(player_centers) > 0:
                    H, status = cv2.findHomography(np.array(corners, dtype=np.float32), np.array(pitch_points, dtype=np.float32))
                    pts = np.array(player_centers, dtype=np.float32).reshape(-1, 1, 2)
                    mapped = cv2.perspectiveTransform(pts, H)
                    mapped_points = [tuple(map(int, pt[0])) for pt in mapped]
                    
                    pitch_width, pitch_height = 800, 500
                    # Clamp positions to pitch boundaries
                    display_positions = []
                    for (x, y) in mapped_points:
                        x = max(0, min(pitch_width - 1, x))
                        y = max(0, min(pitch_height - 1, y))
                        display_positions.append((x, y))
                    
                    # Offside detection using X-axis (horizontal pitch, X is goal-to-goal)
                    # Center of pitch is at x = 400 (pitch_width / 2)
                    center_x = pitch_width // 2
                    
                    team0_indices = [i for i, l in enumerate(final_labels) if l == 0]
                    team1_indices = [i for i, l in enumerate(final_labels) if l == 1]
                    
                    if len(team0_indices) >= 1 and len(team1_indices) >= 2:
                        # Get X positions for both teams (horizontal pitch)
                        team0_x = [display_positions[i][0] for i in team0_indices]
                        team1_x = [display_positions[i][0] for i in team1_indices]
                        
                        # Find defenders' positions (team 1)
                        sorted_team1_x = sorted(team1_x)
                        
                        # Team 1's leftmost defender and second-leftmost (smallest X)
                        leftmost_defender = sorted_team1_x[0]
                        second_leftmost = sorted_team1_x[1] if len(sorted_team1_x) >= 2 else leftmost_defender
                        
                        # Team 1's rightmost defender and second-rightmost (largest X)
                        rightmost_defender = sorted_team1_x[-1]
                        second_rightmost = sorted_team1_x[-2] if len(sorted_team1_x) >= 2 else rightmost_defender
                        
                        # Determine team 1's defensive side (where their goal is)
                        team1_avg_x = np.mean(team1_x)
                        
                        # Check each team 0 player for offside
                        for i in team0_indices:
                            player_x = display_positions[i][0]
                            
                            if team1_avg_x < center_x:
                                # Team 1 is defending the left side (their goal is at left)
                                # Team 0 is attacking left - offside if player is left of second defender
                                if player_x < second_leftmost - 15 and player_x < center_x:
                                    offside_indices.add(i)
                            else:
                                # Team 1 is defending the right side (their goal is at right)
                                # Team 0 is attacking right - offside if player is right of second defender
                                if player_x > second_rightmost + 15 and player_x > center_x:
                                    offside_indices.add(i)
                    
                    if len(team1_indices) >= 1 and len(team0_indices) >= 2:
                        # Get X positions for both teams (horizontal pitch)
                        team0_x = [display_positions[i][0] for i in team0_indices]
                        team1_x = [display_positions[i][0] for i in team1_indices]
                        
                        # Find defenders' positions (team 0)
                        sorted_team0_x = sorted(team0_x)
                        
                        leftmost_defender = sorted_team0_x[0]
                        second_leftmost = sorted_team0_x[1] if len(sorted_team0_x) >= 2 else leftmost_defender
                        
                        rightmost_defender = sorted_team0_x[-1]
                        second_rightmost = sorted_team0_x[-2] if len(sorted_team0_x) >= 2 else rightmost_defender
                        
                        team0_avg_x = np.mean(team0_x)
                        
                        # Check each team 1 player for offside
                        for i in team1_indices:
                            player_x = display_positions[i][0]
                            
                            if team0_avg_x < center_x:
                                # Team 0 is defending the left side
                                # Team 1 is attacking left
                                if player_x < second_leftmost - 15 and player_x < center_x:
                                    offside_indices.add(i)
                            else:
                                # Team 0 is defending the right side
                                # Team 1 is attacking right
                                if player_x > second_rightmost + 15 and player_x > center_x:
                                    offside_indices.add(i)

            for i in range(len(track_ids)):
                if i >= len(tracked_detections):
                    break
                x1, y1, x2, y2 = map(int, tracked_detections.xyxy[i])
                track_id = track_ids[i]
                cluster = final_labels[i] if i < len(final_labels) else 0
                
                # Store team assignment for this track
                if track_id not in track_team_assignments:
                    track_team_assignments[track_id] = cluster
                else:
                    # Use stored team assignment for consistency (with some smoothing)
                    stored_team = track_team_assignments[track_id]
                    if cluster != 2:  # Don't override with referee
                        track_team_assignments[track_id] = cluster
                    cluster = track_team_assignments[track_id]
                
                # # Skip drawing for referees (cluster 2)
                # if cluster == 2:
                #     continue
                
                #If player is offside, use red, else use cluster color
                if i in offside_indices:
                    color = (0, 0, 255)  # Red for offside
                    text = f"#{track_id} OFFSIDE"
                else:
                    color = cluster_colors[cluster % len(cluster_colors)]
                    text = f"#{track_id} T{cluster+1}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw the 2D pitch and mapped player positions
            if len(mapped_points) > 0:
                pitch_width, pitch_height = 800, 500  # horizontal pitch
                pitch_img = np.ones((pitch_height, pitch_width, 3), dtype=np.uint8) * 30  # dark background

                # Draw pitch outline
                cv2.rectangle(pitch_img, (0, 0), (pitch_width-1, pitch_height-1), (0, 255, 0), 2)
                # Draw center line (vertical for horizontal pitch)
                cv2.line(pitch_img, (pitch_width//2, 0), (pitch_width//2, pitch_height), (0, 255, 0), 1)
                # Draw center circle
                cv2.circle(pitch_img, (pitch_width//2, pitch_height//2), 60, (0, 255, 0), 1)

                # Use the display positions directly
                display_points = display_positions
                
                # Draw mapped player positions
                for i, pt in enumerate(display_points):
                    cluster = final_labels[i] if final_labels else 0
                    
                    # Skip referees (cluster 2)
                    if cluster == 2:
                        continue
                    
                    if i in offside_indices:
                        cv2.circle(pitch_img, pt, 10, (0, 0, 255), -1)  # Red for offside
                    else:
                        color = cluster_colors[cluster % len(cluster_colors)]
                        cv2.circle(pitch_img, pt, 10, color, -1)

                cv2.imshow("2D Pitch", pitch_img)

            cv2.imshow("YOLOv11 Detection", frame)
            key = cv2.waitKey(30)
            if key == 27:
                break
            if key == ord('c'):
                cv2.waitKey(0)  # Pause until next key press for debugging
            # Read next frame
            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()
    
    @staticmethod
    def load_or_calibrate_corners(frame, calibration_file):
        """
        Loads saved calibration or prompts user for manual calibration.
        Supports both basic 4-point and enhanced 8-point calibration.
        
        Args:
        - frame: First video frame for calibration
        - calibration_file: JSON file to save/load corner positions
        
        Returns:
        - Tuple: (List of calibration points, use_enhanced flag)
        """
        # Try loading saved calibration
        if os.path.exists(calibration_file):
            try:
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                    corners = [tuple(pt) for pt in data['corners']]
                    use_enhanced = data.get('enhanced', False)
                print(f"Loaded saved calibration from {calibration_file}")
                print(f"  Mode: {'Enhanced (8-point)' if use_enhanced else 'Basic (4-point)'}")
                print("  Delete this file to recalibrate")
                return corners, use_enhanced
            except Exception as e:
                print(f"Could not load calibration: {e}")
        
        # No saved calibration - offer calibration options
        print("\n" + "="*70)
        print("PITCH CALIBRATION REQUIRED")
        print("="*70)
        print("Choose calibration method:")
        print("  1. ENHANCED (Recommended) - 8 points for accurate transformation")
        print("     Better handles perspective distortion and player positioning")
        print("  2. BASIC - 4 corners (simpler but less accurate)")
        print("="*70)
        
        # Ask user for preference
        choice = input("Enter choice (1 for Enhanced, 2 for Basic) [1]: ").strip()
        use_enhanced = choice != "2"
        
        if use_enhanced:
            print("\n" + "="*70)
            print("ENHANCED 8-POINT CALIBRATION")
            print("="*70)
            print("Click 8 points on the pitch in this order:")
            print("  1. Top-Left corner")
            print("  2. Top-Center (middle of top edge)")
            print("  3. Top-Right corner")
            print("  4. Middle-Left (center of left edge)")
            print("  5. Middle-Right (center of right edge)")
            print("  6. Bottom-Left corner")
            print("  7. Bottom-Center (middle of bottom edge)")
            print("  8. Bottom-Right corner")
            print("="*70 + "\n")
            
            corners = VideoProcessing.get_points_from_user(
                frame.copy(), 
                num_points=8, 
                window_name="Enhanced Calibration - Click 8 Points"
            )
            
            if len(corners) != 8:
                print("Enhanced calibration failed, falling back to basic mode")
                use_enhanced = False
        
        if not use_enhanced:
            print("\n" + "="*70)
            print("BASIC 4-POINT CALIBRATION")
            print("="*70)
            print("Click 4 corner points:")
            print("  1. Top-Left corner")
            print("  2. Top-Right corner")
            print("  3. Bottom-Right corner")
            print("  4. Bottom-Left corner")
            print("="*70 + "\n")
            
            corners = VideoProcessing.get_points_from_user(
                frame.copy(), 
                num_points=4, 
                window_name="Basic Calibration - Click 4 Corners"
            )
        
        if len(corners) >= 4:
            # Save calibration for future use
            try:
                with open(calibration_file, 'w') as f:
                    json.dump({
                        'corners': corners,
                        'enhanced': use_enhanced
                    }, f, indent=2)
                print(f" Calibration saved to {calibration_file}")
                print(f"  Delete this file to recalibrate next time")
            except Exception as e:
                print(f" Could not save calibration: {e}")
        
        return corners, use_enhanced
    
    @staticmethod
    def get_enhanced_pitch_points():
        """
        Returns 8 reference points on the 2D pitch for enhanced calibration.
        These correspond to the 8 points the user clicks in enhanced mode.
        
        Points are arranged to match the calibration order:
        Top: TL, TC, TR
        Middle: ML, MR
        Bottom: BL, BC, BR
        
        Horizontal pitch layout (width=800, height=500).
        """
        pitch_width, pitch_height = 800, 500
        return [
            (0, 0),                          # 1. Top-Left
            (pitch_width // 2, 0),           # 2. Top-Center
            (pitch_width, 0),                # 3. Top-Right
            (0, pitch_height // 2),          # 4. Middle-Left
            (pitch_width, pitch_height // 2),# 5. Middle-Right
            (0, pitch_height),               # 6. Bottom-Left
            (pitch_width // 2, pitch_height),# 7. Bottom-Center
            (pitch_width, pitch_height)      # 8. Bottom-Right
        ]
    
    @staticmethod
    def detect_pitch_corners(frame):
        """
        Attempts automatic pitch corner detection using visible pitch lines.
        Less reliable when corners aren't visible - use load_or_calibrate_corners() instead.
        
        Returns the four corners as a list of (x, y) tuples or None if detection fails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # Detect lines with lower thresholds to catch more pitch markings
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=150, maxLineGap=50)
        if lines is None:
            return None

        # Collect all line endpoints
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append((x1, y1))
            points.append((x2, y2))

        if len(points) < 4:
            return None
            
        points = np.array(points)
        hull = cv2.convexHull(points)
        
        if len(hull) < 4:
            return None

        # Approximate hull to 4 points (corners)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) != 4:
            # Fallback: take 4 extreme points
            try:
                from scipy.spatial import ConvexHull
                hull_indices = ConvexHull(points).vertices
                if len(hull_indices) >= 4:
                    approx = points[hull_indices][:4]
                else:
                    return None
            except:
                return None

        corners = [tuple(pt[0]) if isinstance(pt[0], np.ndarray) else tuple(pt) for pt in approx]
        return corners

    @staticmethod
    def sort_corners(pts):
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return [
            tuple(pts[np.argmin(s)]),      # top-left
            tuple(pts[np.argmin(diff)]),   # top-right
            tuple(pts[np.argmax(s)]),      # bottom-right
            tuple(pts[np.argmax(diff)])    # bottom-left
        ]
    
    @staticmethod
    def color_based_team_assignment_with_referee(avg_hsvs, kmeans_labels, team1_hsv, team2_hsv, referee_hsv, track_ids=None, threshold=15):
        """
        Assigns players to teams or referee based on color similarity and k-means clustering.

        Args:
        - avg_hsvs: List of average HSV colors for each player.
        - kmeans_labels: Result from k-means (0, 1, or 2 for each player).
        - team1_hsv, team2_hsv: Known team colors in HSV.
        - referee_hsv: Known referee color in HSV (typically black/dark).
        - track_ids: List of tracker IDs for each player (for console output).
        - threshold: Distance threshold to decide if a player is closer to one team color.

        Returns:
        - List of final team assignments: 0=team1, 1=team2, 2=referee.
        """
        final_labels = []
        for i, player_hsv in enumerate(avg_hsvs):
            # Distance to known team colors and referee
            d1 = np.linalg.norm(player_hsv - team1_hsv)
            d2 = np.linalg.norm(player_hsv - team2_hsv)
            d_ref = np.linalg.norm(player_hsv - referee_hsv)
            
            # Check for referee first (low saturation + low-mid value = black/dark clothing)
            # Referees typically have very low saturation regardless of value
            saturation = player_hsv[1]
            value = player_hsv[2]
            is_dark = saturation < 100 and value < 130
            
            # Strong referee indicator: very low saturation (grayscale/black)
            is_very_dark = saturation < 60
            
            if is_very_dark:
                # Very confident this is referee - prioritize even if distance isn't closest
                assigned_label = 2  # Referee
                assigned_to = "referee (very dark)"
            elif is_dark and d_ref < max(d1, d2) * 0.8:  # If somewhat dark and closer to black than far from teams
                assigned_label = 2  # Referee
                assigned_to = "referee (dark)"
            elif d1 < d2:
                assigned_label = 0  # Team 1
                assigned_to = "team 1"
            else:
                assigned_label = 1  # Team 2
                assigned_to = "team 2"

            # Debug: print HSV and distances (use track_id for consistent numbering with video)
            display_id = track_ids[i] if track_ids is not None else i
            print(f"Player #{display_id}: HSV={player_hsv}, Sat={saturation:.0f}, Val={value:.0f}, d1={d1:.1f}, d2={d2:.1f}, d_ref={d_ref:.1f} -> {assigned_to}")

            # Use color-based assignment if clear difference, otherwise use k-means
            if is_very_dark or is_dark or abs(d1 - d2) > threshold:
                final_labels.append(assigned_label)
            else:
                final_labels.append(kmeans_labels[i])
        return final_labels
    
    @staticmethod
    def color_based_team_assignment(avg_hsvs, kmeans_labels, team1_hsv, team2_hsv, threshold=15):
        """
        Assigns players to teams based on color similarity and k-means clustering results.

        Args:
        - avg_hsvs: List of average HSV colors for each player.
        - kmeans_labels: Result from k-means (0 or 1 for each player).
        - team1_hsv, team2_hsv: Known team colors in HSV.
        - threshold: Distance threshold to decide if a player is closer to one team color.

        Returns:
        - List of final team assignments for each player.
        """
        final_labels = []
        for i, player_hsv in enumerate(avg_hsvs):
            # Distance to known team colors
            d1 = np.linalg.norm(player_hsv - team1_hsv)
            d2 = np.linalg.norm(player_hsv - team2_hsv)
            color_label = 0 if d1 < d2 else 1

            # Debug: print HSV and distances
            print(f"Player {i}: avg_hsv={player_hsv}, d1={d1}, d2={d2}, assigned to {'team 1' if d1 < d2 else 'team 2'}")

            if abs(d1 - d2) < threshold:
                final_labels.append(kmeans_labels[i])
            else:
                final_labels.append(color_label)
        return final_labels
    
        
    def get_points_from_user(image, num_points=4, window_name="Select Points"):
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
                points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(window_name, image)

        clone = image.copy()
        cv2.imshow(window_name, clone)
        cv2.setMouseCallback(window_name, mouse_callback)

        print(f"Please click {num_points} points on the image window...")
        while len(points) < num_points:
            cv2.waitKey(1)
        cv2.destroyWindow(window_name)
        return points