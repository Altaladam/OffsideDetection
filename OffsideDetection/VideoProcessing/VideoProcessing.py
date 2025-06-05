import cv2, pafy, yt_dlp, os
from cv2.gapi import video
from pytube import YouTube
from ultralytics import YOLO
import neptune
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from VideoProcessing.yolo_segmentation import YOLOSegmentation
from VideoProcessing.functions import get_average_color, classify_bgr_color


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
        
    def init_run(tags=None):
        run = neptune.init_run(
        project="bexgboost/project",
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        tags=tags,
        )    
        return run
        
    def YOLO():
        # url   = "https://www.youtube.com/watch?v=3N7BkyuEBAw&ab_channel=HashtagUnited&t=6090s"
        # def is_video_downloaded(url):
        #     return os.path.exists(f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4")
        # if not is_video_downloaded(url):
        #     video = yt_dlp.YoutubeDL().download(url)
        # else:
        video = f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4"

        #video = f"vid.mov"
        
        #video = f"fm_test.mkv"
        cap = cv2.VideoCapture(video)
        
        ys = YOLOSegmentation("yolov8m-seg.pt")
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        new_points = []

        colors = []

        player_coords = []

        # INPUT TEAM COLORS (BGR)
        team1_bgr = [255, 255, 96]
        team2_bgr = [255, 110, 126]
        #team1_bgr = [0, 0, 159]
        #team2_bgr = [0, 145, 129]

        # INPUT PERSPECTIVE COORDINATES ON ORIGINAL IMAGE (TL, BL, TR, BR)
        og_perspective_coords = [[782, 349], [1585, 343], [96, 798], [1559, 801]]
        # INPUT PERSPECTIVE COORDINATES ON NEW IMAGE (TL, BL, TR, BR)
        new_perspective_coords = [[67, 31], [305, 32], [69, 652], [306, 652]]

        # Perspective transform function (pass in a point) (returns a point)
        def perspective_transform(player, team, original, new):
            tl, bl, tr, br = original
            tl2, bl2, tr2, br2 = new
            
            p = player

            pts1 = np.float32([tl,bl,tr,br])
            pts2 = np.float32([tl2,bl2,tr2,br2])

            pts3 = np.float32([p])

            M = cv2.getPerspectiveTransform(pts1,pts2)

            pts3o=cv2.perspectiveTransform(pts3[None, :, :], M)
            x = int(pts3o[0][0][0])
            y = int(pts3o[0][0][1])
            new_p = (x,y)

            # Place transformed point for each player on dst
            if(team == "group1"):
                cv2.circle(dst,new_p,10,team1_bgr,-1)
                new_points.append(new_p)
            if(team == "group2"):
                cv2.circle(dst,new_p,10,team2_bgr,-1)
                new_points.append(new_p)

            cv2.imshow('Top View', dst)

        # Loop through each frame
        while True:
            # Video frame = frame
            ret, frame = cap.read()

            # 2D image = dst
            dst = cv2.imread("dst.png")

            if not ret:
                break

            # Copy of frame
            frame2 = np.array(frame)

            # Detect objects
            bboxes, classes, segmentations, scores = ys.detect(frame)

            player_coords.clear()
            colors.clear()
            new_points.clear()

            # Loop through each object
            for index, (bbox, class_id, seg, score) in enumerate(zip(bboxes, classes, segmentations, scores)):
                # If object is a player
                if class_id == 0:
                    # Set corner coordinates for bounding box around player
                    (x, y, x2, y2) = bbox
                    
                    # Draw segmentation around player
                    if len(seg) != 0:
                        minX = min(seg, key=itemgetter(0))[0]
                        maxX = max(seg, key=itemgetter(0))[0]
                        maxY = max(seg, key=itemgetter(1))[1]

                        # Create smaller rectangle around player to use for color detection
                        distLeft = int(abs(seg[0][0] - minX))
                        distRight = int(abs(seg[0][0] - maxX))

                        # Get smaller box points around player for detecting color
                        newX = int((x2 - x)/3 + x)
                        newY = int((y2 - y)/5 + y)
                        newX2 = int(2*(x2 - x)/3 + x)
                        newY2 = int(2*(y2 - y)/5 + y)

                        # Shift color detection box based on player orientation
                        if(distRight > distLeft):
                            if(distLeft == 0):
                                distLeft+= 1
                            # Shift left
                            newX = int(newX - ((distRight)/distLeft)/1.5)
                            newX2 = int(newX2 - ((distRight)/distLeft)/1.5)
                        else:
                            # Shift right
                            if(distRight == 0):
                                distRight+= 1
                            newX = int(newX + ((distLeft)/distRight)*1.5)
                            newX2 = int(newX2 + ((distLeft)/distRight)*1.5)

                        # Define smaller rectangle around player to use for color detection
                        roi = frame2[newY:newY2, newX:newX2]

                        # Get average color of smaller rectangle
                        dominant_color = get_average_color(roi)
                        cv2.rectangle(frame, (newX, newY), (newX2, newY2), dominant_color, 2)

                        team = classify_bgr_color(dominant_color, team1_bgr, team2_bgr)

                        if(team == "group1"):
                            cv2.putText(frame, "Team 1", (x, y-5), font, 1, team1_bgr, 3, cv2.LINE_AA)
                            
                            # Draw segmentation with the color of the dominant color of the player
                            cv2.polylines(frame, [seg], True, team1_bgr, 3)
                            cv2.circle(frame,(minX, maxY),5,team1_bgr,-1)
                        if(team == "group2"):
                            cv2.putText(frame, "Team 2", (x, y-5), font, 1, team2_bgr, 3, cv2.LINE_AA)

                            # Draw segmentation with the color of the dominant color of the player
                            cv2.polylines(frame, [seg], True, team2_bgr, 3)
                            cv2.circle(frame,(minX, maxY),5,team2_bgr,-1)

                # Perspective transform for each player
                perspective_transform([minX, maxY], team, og_perspective_coords, new_perspective_coords)
            

            # Find furthest player and place vertical line
            max_point_X, max_point_Y = min(new_points, key=itemgetter(0))[0], min(new_points, key=itemgetter(0))[1]
            cv2.circle(dst, (max_point_X, max_point_Y), 10, (0,255,255), 2)
            cv2.line(dst, (max_point_X, 0), (max_point_X, 1035), (0,255,255), 2)

            # Show images
            cv2.imshow("Img", frame)
            #cv2.imshow("Top View", dst)

            
            # Space to move forward a frame
            key = cv2.waitKey(20)
            # Esc to exit
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def YOLO2():
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
        model = YOLO(MODEL_NAME)
        model.to('cuda')
        CONF_THRESHOLD = 0.3

        # Get the first frame for automatic pitch corner detection
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video.")
            return

        # Automatically detect pitch corners
        corners = VideoProcessing.detect_pitch_corners(frame)
        if corners is None or len(corners) != 4:
            print("Could not automatically detect pitch corners.")
            return
        print("Detected pitch corners:", corners)
        
        # Manually select 4 points
        # frame_points = VideoProcessing.get_points_from_user(frame, num_points=4)
        # print("Selected points:", frame_points)
        
        # 1. User clicks 4 points on the input frame
        # frame_points = VideoProcessing.get_points_from_user(frame, num_points=4, window_name="Input Frame")
        # print("Selected points in frame:", frame_points)

        # # 2. User clicks 4 corresponding points on the pitch map
        # pitch_width, pitch_height = 500, 800  # match your output 2D pitch
        # pitch_map = np.ones((pitch_height, pitch_width, 3), dtype=np.uint8) * 30  # dark background

        # # Draw pitch outline
        # cv2.rectangle(pitch_map, (0, 0), (pitch_width-1, pitch_height-1), (0, 255, 0), 2)
        # # Draw center line
        # cv2.line(pitch_map, (0, pitch_height//2), (pitch_width, pitch_height//2), (0, 255, 0), 1)
        # # Draw center circle
        # cv2.circle(pitch_map, (pitch_width//2, pitch_height//2), 60, (0, 255, 0), 1)

        # pitch_points = VideoProcessing.get_points_from_user(pitch_map, num_points=4, window_name="Pitch Map")
        # print("Selected points on pitch map:", pitch_points)

        # # 3. Compute homography
        # H, status = cv2.findHomography(np.array(frame_points, dtype=np.float32),
        #                             np.array(pitch_points, dtype=np.float32))
        # print("Homography matrix:\n", H)

        # Example 2D pitch points (adjust to pitch template size)
        pitch_points = [(0, 0), (500, 0), (500, 800), (0, 800)]
        
        #  # Use center circle detection
        # center_circle = VideoProcessing.detect_center_circle(frame)
        # if center_circle is None:
        #     print("Could not automatically detect center circle.")
        #     return
        # center_x, center_y, radius = center_circle
        # print(f"Detected center circle: center=({center_x}, {center_y}), radius={radius}")

        # Define known team colors in HSV
        
        #team1_hsv = np.array([251, 157, 221])  # colors for vid.mov
        #team2_hsv = np.array([13, 28, 103])
        #team1_hsv = np.array([255, 255, 96])    # colors for FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4
        #team2_hsv = np.array([255, 110, 126])
        team1_hsv = np.array([45, 120, 170])    # colors for FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4
        team2_hsv = np.array([90, 100, 140])



        while True:
            if not ret:
                break

            results = model(frame)
            boxes = results[0].boxes
            mask = boxes.conf.cpu().numpy() > CONF_THRESHOLD
            filtered_boxes = boxes[mask]

            colors = []
            avg_hsvs = []
            player_centers = []
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                h, w = y2 - y1, x2 - x1
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
                print(f"Team sample avg_hsv: {avg_hsv}")
                avg_hsvs.append(avg_hsv)
                player_centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

            offside_indices = set()
            mapped_points = []
            final_labels = []
            if len(colors) >= 2:
                Z = np.float32(colors)
                K = 2  # number of clusters (e.g., 2 teams)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret_km, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                cluster_colors = [(0, 255, 255), (255, 0, 255)]

                # Combine k-means and color-based assignment
                final_labels = VideoProcessing.color_based_team_assignment(
                    avg_hsvs, label.flatten(), team1_hsv, team2_hsv, threshold=15
                )

                # Homography: Map player centers to 2D pitch
                if len(player_centers) > 0:
                    H, status = cv2.findHomography(np.array(corners, dtype=np.float32), np.array(pitch_points, dtype=np.float32))
                    pts = np.array(player_centers, dtype=np.float32).reshape(-1, 1, 2)
                    mapped = cv2.perspectiveTransform(pts, H)
                    mapped_points = [tuple(map(int, pt[0])) for pt in mapped]

                    # # Offside detection logic
                    # # Assume final_labels: 0 is attackers, 1 is defenders
                    # for team in range(K):
                    #     team_indices = [i for i, l in enumerate(final_labels) if l == team]
                    #     if len(team_indices) == 0:
                    #         continue
                    #     # For simplicity, assume team 0 is attackers, team 1 is defenders
                    #     # Find the max y (closest to goal line) among defenders
                    #     if team == 1:
                    #         defender_ys = [mapped_points[i][1] for i in team_indices]
                    #         if len(defender_ys) >= 2:
                    #             sorted_defenders = sorted(defender_ys)
                    #             offside_line = sorted_defenders[-2]  # second last defender
                    #             # Mark attackers beyond this line as offside
                    #             for i, l in enumerate(final_labels):
                    #                 if l == 0 and mapped_points[i][1] > offside_line:
                    #                     offside_indices.add(i)

            for i, box in enumerate(filtered_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cluster = final_labels[i]
                # If player is offside, use red, else use cluster color
                # if i in offside_indices:
                #     color = (0, 0, 255)  # Red for offside
                #     text = "OFFSIDE"
                # else:
                color = cluster_colors[cluster % len(cluster_colors)]
                text = f"Team {cluster+1}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

            # Draw the 2D pitch and mapped player positions
            if len(mapped_points) > 0:
                pitch_width, pitch_height = 500, 800  # match pitch_points
                pitch_img = np.ones((pitch_height, pitch_width, 3), dtype=np.uint8) * 30  # dark background

                # Draw pitch outline
                cv2.rectangle(pitch_img, (0, 0), (pitch_width-1, pitch_height-1), (0, 255, 0), 2)
                # Draw center line
                cv2.line(pitch_img, (0, pitch_height//2), (pitch_width, pitch_height//2), (0, 255, 0), 1)
                # Draw center circle
                cv2.circle(pitch_img, (pitch_width//2, pitch_height//2), 60, (0, 255, 0), 1)

                # Draw mapped player positions
                for i, pt in enumerate(mapped_points):
                    if i in offside_indices:
                        cv2.circle(pitch_img, pt, 10, (0, 0, 255), -1)  # Red for offside
                    else:
                        cluster = final_labels[i] if final_labels else 0
                        color = cluster_colors[cluster % len(cluster_colors)]
                        cv2.circle(pitch_img, pt, 10, color, -1)

                cv2.imshow("2D Pitch", pitch_img)

            cv2.imshow("YOLOv11 Detection", frame)
            key = cv2.waitKey(30)
            if key == 27:
                break
            
            # Read next frame
            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()
    
    @staticmethod
    def detect_pitch_corners(frame):
        """
        Automatically detects four pitch corners in a football field image.
        Returns the four corners as a list of (x, y) tuples: [top-left, top-right, bottom-right, bottom-left]
        """
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=200, maxLineGap=50)
        if lines is None:
            return None

        # Collect all line endpoints
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append((x1, y1))
            points.append((x2, y2))

        points = np.array(points)

        hull = cv2.convexHull(points)
        if len(hull) < 4:
            return None

        # Approximate hull to 4 points (corners)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) != 4:
            # fallback: take 4 farthest points
            from scipy.spatial import ConvexHull
            hull_indices = ConvexHull(points).vertices
            approx = points[hull_indices][:4]

        corners = [tuple(pt[0]) if isinstance(pt[0], np.ndarray) else tuple(pt) for pt in approx]
        # Optionally, sort corners (top-left, top-right, bottom-right, bottom-left)
        # for x, y in corners:
        #     cv2.circle(frame, (x, y), 10, (0, 0, 255), 3)
        # cv2.imshow("Corners Debug", frame)
        # cv2.waitKey(1)
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
    
    @staticmethod
    def detect_center_circle(frame):
        """
        Detects the center circle of a football pitch in the given frame.
        Returns (center_x, center_y, radius) if found, else None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 2)
        edges = cv2.Canny(blur, 50, 150)

        # Hough Circle Transform
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=30,
            maxRadius=120
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Take the largest circle (most likely the center circle)
            largest = max(circles[0, :], key=lambda c: c[2])
            center_x, center_y, radius = largest
            # Draw for debug
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)
            cv2.imshow("Center Circle Debug", frame)
            cv2.waitKey(1)
            return (center_x, center_y, radius)
        else:
            print("No center circle detected.")
            return None
        
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