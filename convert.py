import os
import json
from PIL import Image

def convert_yolo_to_coco(yolo_dir, output_json):
    yolo_images_dir = os.path.join(yolo_dir, "images")
    yolo_labels_dir = os.path.join(yolo_dir, "labels")

    images = []
    annotations = []
    categories = []
    category_set = set()
    annotation_id = 1

    for idx, label_file in enumerate(os.listdir(yolo_labels_dir)):
        if not label_file.endswith('.txt'):
            continue

        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(yolo_images_dir, image_file)
        label_path = os.path.join(yolo_labels_dir, label_file)

        with Image.open(image_path) as img:
            width, height = img.size

        image_info = {
            "id": idx,
            "file_name": image_file,
            "width": width,
            "height": height
        }
        images.append(image_info)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                bbox_width = float(parts[3]) * width
                bbox_height = float(parts[4]) * height

                x_min = x_center - bbox_width / 2
                y_min = y_center - bbox_height / 2

                if class_id not in category_set:
                    categories.append({
                        "id": class_id,
                        "name": str(class_id), 
                        "supercategory": "none"
                    })
                    category_set.add(class_id)

                annotation_info = {
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                }
                annotations.append(annotation_info)
                annotation_id += 1

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    output_dir = os.path.dirname(output_json)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=4)

if __name__ == "__main__":
    yolo_dir = r"C:\\Users\\imanj\\Desktop\\codes\\python-converter\\yolo-data" 
    output_json = r"C:\\Users\\imanj\\Desktop\\codes\\python-converter\\output\\coco_dataset.json" 

    convert_yolo_to_coco(yolo_dir, output_json)
    print(f"COCO dataset JSON saved to {output_json}")
