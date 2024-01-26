import cv2

def get_model(configPath, weightsPath):
    # load our YOLO object detector trained on the COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net
