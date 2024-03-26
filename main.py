import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt


#test_image = Image.open(frame[y2:y1, x2:x1]).convert('RGB')

model = YOLO('yolov8x.pt')

video_path = r"C:\Users\user\Downloads\Test.mp4"
cap = cv2.VideoCapture(video_path)

points = []
test_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def draw_polygon(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 3:
            cv2.polylines(img, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
            # points = []

saved_model_path = 'resnet_model_train_97_val_95.pth'
loaded_model = models.resnet50(pretrained=False)
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = torch.nn.Linear(num_ftrs, 2)
loaded_model.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
loaded_model.eval()
            
unique_id = None
counter = 0            
all_count = 0
frame_count = 0
c = 0
pred = [0, 0, 0]


img = np.ones((720, 1280 , 3), np.uint8) * 255
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('Draw Polygon')
cv2.setMouseCallback('Draw Polygon', draw_polygon)

while cap.isOpened():
    success, frame = cap.read()
    # frame = cv2.resize(frame, (1500, 800))
    frame_count += 1
    if success:
        results = model.track(frame, persist=True, classes=2, conf=0.1, iou=0.2, imgsz=(320,320))
        for out in results[0].boxes.data.tolist():
            try:
                x1, y1, x2, y2, Id, score, class_id = out
                center = ((x1+x2) / 2, (y1+y2) / 2)
                # Id = out[4]
                
                # if int(Id) == 3:
                check = cv2.pointPolygonTest(np.array(points), center, measureDist=False)
                if check == 1:
                    if Id != unique_id:
                        print("unique_id take")
                        all_count += 1
                        unique_id = Id
                        c = 0
                        pred = [0, 0, 0]
                    elif Id == unique_id:
                        print("unique_id old",c,pred[-3:],counter, frame_count)
                        
                        if frame_count % 100 == 0:
                            print("unique_id old",c,len(pred),pred[-3:],counter, frame_count)
                            if c < 3:
                                print("unique_id old input img",c,pred[-3:],counter, frame_count)
                                
                                pil_image = Image.fromarray(frame[int(y1):int(y2), int(x1):int(x2)])
                                input_image = test_transform(pil_image).unsqueeze(0)
                                
                                with torch.no_grad():
                                    output = loaded_model(input_image)
                                    
                                _, predicted = torch.max(output, 1)
                                predicted_class = predicted.item()
                                pred.append(predicted_class)
                                print("predicted_class == ",predicted_class)
                                
                                if sum(pred[-3:]) >= 2:
                                    counter += 1
                                    c = 3
                                elif sum(pred[-3:]) < 2:
                                    c = 0
            except:
                continue
        annotated_frame = results[0].plot()
        annotated_frame = cv2.bitwise_and(annotated_frame, img)
        
        cv2.imshow("Draw Polygon", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()