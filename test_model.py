from ultralytics import YOLO
from PIL import Image
img_path = 'test1.jpeg'  
img = Image.open(img_path)
model = YOLO('./runs/detect/train6/weights/best.pt')
results = model(['test1.jpeg','test2.jpeg'],conf=0.5,save=True,boxes=True,classes=[0,1,2,3],show_labels=True,save_conf=True,augment=True,imgsz=640)
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  
    print('probs==>',result)
# result.show()
# result.save('test1.jpg')
# print('result',img)
# img_with_results = results.show()
# img_with_results.save('test1.jpg') 