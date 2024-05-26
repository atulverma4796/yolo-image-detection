from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
# model_path = './runs/segment/train/weights/best.pt'
model_path = 'goal_score.pt'

# class_name_dict = {0: 'football', 1: "goal_post", 2: 'team_dark', 3: 'team_light'}
class_name_dict = {0: 'goal_scored'}

path = './videos/7.mp4'
# path = './test_dataset/test28.webp'
# path = './test_dataset/test17.jpg'

file_extension = Path(path).suffix

model = YOLO(model_path)

if file_extension.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
    img = Image.open(path)
    img_gray = img.convert('L')
    results = model.predict(img_gray)

    if results is not None and len(results) > 0:
        result = results[0]
        masks = result.masks

        if masks is not None:
            fnt = ImageFont.truetype("FreeMonoBold.otf", 30)
            draw = ImageDraw.Draw(img)
            clss = result.boxes.cls.cpu().tolist()
            conf = result.boxes.conf.cpu().tolist()
            keypoint = result.keypoints
            football_polygons = []
            goal_post_polygons = []
            for i, mask in enumerate(masks):
                # print("mask",mask)
                polygon = mask.xy[0]
                x, y = polygon[0]
                y -= 23
                class_index = clss[i]
                class_name = class_name_dict.get(class_index)
                confidence_score = conf[i]
               
                polygon_coords = [(x, y) for x, y in polygon]
                polygon_shapely = Polygon(polygon_coords)

                if class_name == "football":
                    football_polygons.append(polygon_shapely)
                elif class_name == "goal_post":
                    goal_post_polygons.append(polygon_shapely)
                    
                if class_name == "team_dark":
                    draw.polygon(polygon, outline=(255, 255, 0), width=3)
                    draw.text((x, y), f"Team-A, Conf: {confidence_score:.2f}", font=fnt, fill=(0, 0, 255))
                elif class_name == "team_light":
                    draw.polygon(polygon, outline=(0, 255, 0), width=3)
                    draw.text((x, y), f"Team-B, Conf: {confidence_score:.2f}", font=fnt, fill=(0, 0, 255))
                else:
                    draw.polygon(polygon, outline=(255, 0, 0), width=3)
                    draw.text((x, y), f"{class_name}, Conf: {confidence_score:.2f}", font=fnt, fill=(0, 0, 255))
            goal_scored = False
            for football_polygon in football_polygons:
                for goal_post_polygon in goal_post_polygons:
                    if goal_post_polygon.contains(football_polygon):
                        print("Goal scored! Football is inside the goal post.")
                        goal_scored = True
                    break
            if goal_scored:
                draw.text((10, 10), "Goal Scored!", font=fnt, fill=(0, 255, 0))
            img.show()
        else:
            print("No masks detected in the results.")
    else:
        print("No results from the model.")

if file_extension.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (frame_width, frame_height), isColor=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        # input_image = rgb_frame[None, ...]
        # input_image = input_image[..., ::-1].transpose(0, 3, 1, 2)
        input_image = np.expand_dims(rgb_frame, axis=0)
        # if input_image.size < 9:  # Choose a threshold for valid dimensions
        #     print("Input image has invalid dimensions. Skipping prediction.")
        #     continue
        # print('gray frame',gray_frame)
        # cv2.imshow("Live", gray_frame) 
        results = model.predict(frame)

        if results is not None and len(results) > 0:
            result = results[0]
            masks = result.masks

            if masks is not None:
                clss = result.boxes.cls.cpu().tolist()
                img = frame
                fnt = cv2.FONT_HERSHEY_SIMPLEX
                goal_scored = False

                football_polygons = []
                goal_post_polygons = []
                for i, mask in enumerate(masks):
                    polygon = mask.xy[0]
                    class_index = clss[i]
                    class_name = class_name_dict.get(class_index)
                    if class_name == "team_dark":
                        color = (255, 255, 0)  # Yellow color
                        new_class_name = "Team-A"
                    elif class_name == "team_light":
                        color = (0, 255, 0)  # Green color
                        new_class_name = "Team-B"
                    else:
                        color = (255, 0, 0)  # Red color
                        new_class_name = class_name

                    for i in range(len(polygon) - 1):
                        cv2.line(img, tuple(map(int, polygon[i])), tuple(map(int, polygon[i + 1])), color, 3)
                    cv2.line(img, tuple(map(int, polygon[-1])), tuple(map(int, polygon[0])), color, 3)
                    cv2.putText(img, new_class_name, (int(polygon[0][0]), int(polygon[0][1]) - 10), fnt, 2.0, (0, 0, 255), 4)
                    if class_name == "football":
                        football_polygons.append(polygon)
                    elif class_name == "goal_post":
                        goal_post_polygons.append(polygon)
                for football_polygon_coords in football_polygons:
                    football_polygon = Polygon(football_polygon_coords)
                    for goal_post_polygon_coords in goal_post_polygons:
                        goal_post_polygon = Polygon(goal_post_polygon_coords)
                        if goal_post_polygon.contains(football_polygon):
                            print("Goal scored! Football is inside the goal post!!!.")
                            goal_scored = True
                            break
                if goal_scored:
                    cv2.putText(img, "Goal Scored!", (50, 150), fnt, 2.0, (0, 0, 0), 4)

                cv2.imshow('Video', img)
                out.write(img)
            else:
                print("No masks detected in the results.")
        else:
            print("No results from the model.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


       