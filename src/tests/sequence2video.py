import cv2 as cv
from sys import argv
from os import listdir


if __name__ == "__main__":
    folder = argv[1]
    frames = sorted(f for f in listdir(folder) if f.lower().endswith(".png"))
    frames = [cv.imread(f"{folder}/{f}", cv.IMREAD_COLOR) for f in frames]
    video = cv.VideoWriter(f"{folder}/video.mp4", cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, frames[0].shape[:2])
    for i, frame in enumerate(frames):
        print(f"Frame {i}")
        video.write(frame)
