import cv2, pafy, yt_dlp, os
from cv2.gapi import video
from pytube import YouTube
import numpy as np


class VideoProcessing():

    def read():
        url   = "https://www.youtube.com/watch?v=3N7BkyuEBAw&ab_channel=HashtagUnited"
        def is_video_downloaded(url):
            return os.path.exists(f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4")
        if not is_video_downloaded(url):
            video = yt_dlp.YoutubeDL().download(url)
        else:
            video = f"FULL MATCH! - White Ensign vs Hashtag United [3N7BkyuEBAw].mp4"

        capture = cv2.VideoCapture(video)
        
        backSub = cv2.createBackgroundSubtractorKNN() 
        
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while(capture.isOpened()):
            check, frame = capture.read()
            if check == True:
                fgMask = backSub.apply(frame)
                
                kernel = np.ones((5,5),np.uint8)
                opening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
                
                colored = cv2.bitwise_and(frame, frame, mask = opening);
                cv2.imshow('frame',frame)
                cv2.imshow('FG Mask', colored)
                cv2.waitKey(30)
                            
            else:
                break

        capture.release()
        cv2.destroyAllWindows()