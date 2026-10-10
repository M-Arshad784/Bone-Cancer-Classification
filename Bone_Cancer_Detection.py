from unittest import result

import cv2
import cvzone
from ultralytics import YOLO

img = cv2.imread('c2.jpg')
img = cv2.resize(img, (500, 600))

model = YOLO('best.pt')
results = model(img)

for result in results:

    cls = result.probs.top1
    conf = result.probs.top1conf
    name = model.names[cls]

    print(f'Bone Cancer {name}')
    print(f'Confidence {conf: .2f}')
    cvzone.putTextRect(img, f'{name}, {conf: .2f}', pos=(20, 30), scale=1.5, thickness=2, colorT=(0, 0, 0))

cv2.imshow('Bonce Cancer', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
