import os
import json
import argparse
import shutil
import random
import datetime
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Convert custom dataset to COCO format.")
    parser.add_argument("--input", required=True, help="Input root directory containing annotations and images folders.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--split", type=float, nargs=3, default=[8, 1, 1], help="Split ratio for train/valid/test (e.g., 8 1 1).")
    return parser.parse_args()

def is_box_inside(inner_box, outer_box):
    """
    Check if inner_box is 100% inside outer_box.
    Format: [xmin, ymin, xmax, ymax]
    """
    ix_min, iy_min, ix_max, iy_max = inner_box
    ox_min, oy_min, ox_max, oy_max = outer_box

    return (ix_min >= ox_min and iy_min >= oy_min and 
            ix_max <= ox_max and iy_max <= oy_max)

def load_symptoms_map(symptoms_path):
    with open(symptoms_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    label_map = data.get("label_map", {})
    categories = []
    # Create reverse map: label_string -> id_int
    str_to_id = {}
    
    for key_id, info in label_map.items():
        int_id = int(key_id)
        name = info.get("en", "unknown")
        supercat = info.get("zh", "unknown")
        
        categories.append({
            "id": int_id,
            "name": name,
            "supercategory": supercat
        })
        str_to_id[name] = int_id
        
    return categories, str_to_id

def process_single_pair(json_path, img_dir, global_img_id, str_to_id):
    """
    Process a single JSON/Image pair and return COCO format dicts.
    Returns: (image_info, annotations_list, success_bool, src_img_path)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None, [], False, None

    # 1. Skip if comments exists and len > 0
    if "comments" in data and isinstance(data["comments"], list) and len(data["comments"]) > 0:
        return None, [], False, None

    img_filename = data.get("image_filename")
    if not img_filename:
        return None, [], False, None
        
    img_path = os.path.join(img_dir, img_filename)
    if not os.path.exists(img_path):
        # Try finding it recursively or handle error? Assuming flat structure as per prompt
        print(f"Warning: Image {img_filename} not found in {img_dir}. Skipping.")
        return None, [], False, None

    # Get actual image dimensions
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None, [], False, None

    # Process Metadata
    date_captured = data.get("last_modified_at", "2025-09-01T00:00:00+00:00")
    
    # Custom fields
    custom_fields = {k: data.get(k, None) for k in [
        "generated_by", "general_editor", "expert_editor", 
        "overall", "global_causes_zh", "global_treatments_zh"
    ]}

    detections = data.get("detections", [])
    if detections is None:
        detections = []

    # 2. Logic: Remove "healthy_region" if it contains a wound
    # First, separate boxes
    healthy_indices = []
    wound_boxes = []

    for idx, det in enumerate(detections):
        label = det.get("label")
        box = det.get("box_xyxy")
        if label == "healthy_region":
            healthy_indices.append(idx)
        elif box:
            wound_boxes.append(box)

    indices_to_remove = set()
    for h_idx in healthy_indices:
        h_box = detections[h_idx].get("box_xyxy")
        if not h_box: continue
        
        # Check if ANY wound box is inside this healthy box
        for w_box in wound_boxes:
            if is_box_inside(w_box, h_box):
                indices_to_remove.add(h_idx)
                break
    
    valid_detections = [d for i, d in enumerate(detections) if i not in indices_to_remove]

    # 3. Logic: isHealthy flag
    # If detections is empty OR only contains "healthy_region"
    non_healthy_count = sum(1 for d in valid_detections if d.get("label") != "healthy_region")
    is_healthy = (len(valid_detections) == 0) or (non_healthy_count == 0)

    # Build Image Info
    image_info = {
        "id": global_img_id,
        "file_name": img_filename,
        "width": width,
        "height": height,
        "date_captured": date_captured,
        "license": 1,
        "isHealthy": is_healthy,
        **custom_fields # Unpack custom fields
    }

    # Build Annotations
    coco_annotations = []
    for det in valid_detections:
        label_str = det.get("label")
        if label_str not in str_to_id:
            # Handle unknown label or skip? Assuming label map is complete.
            continue
            
        cat_id = str_to_id[label_str]
        bbox_xyxy = det.get("box_xyxy") # [xmin, ymin, xmax, ymax]
        
        if not bbox_xyxy or len(bbox_xyxy) != 4:
             continue
        
        # Convert to XYWH
        x, y = bbox_xyxy[0], bbox_xyxy[1]
        w = bbox_xyxy[2] - bbox_xyxy[0]
        h = bbox_xyxy[3] - bbox_xyxy[1]
        
        evidence_zh = det.get("evidence_zh", "")

        anno = {
            "image_id": global_img_id,
            "category_id": cat_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation": [], # Bbox only
            "iscrowd": 0,
            "evidence_zh": evidence_zh
        }
        coco_annotations.append(anno)

    return image_info, coco_annotations, True, img_path

def main():
    args = parse_args()
    
    # Validate split
    split_ratios = args.split
    total_ratio = sum(split_ratios)
    if total_ratio == 0:
        raise ValueError("Split ratios cannot be all zero.")
    # Normalize ratios
    split_ratios = [x / total_ratio for x in split_ratios]

    # Setup directories
    symptoms_path = os.path.join(args.input, "symptoms.json")
    if not os.path.exists(symptoms_path):
        raise FileNotFoundError(f"symptoms.json not found at {symptoms_path}")

    # Load Categories
    categories, str_to_id = load_symptoms_map(symptoms_path)

    # Gather pairs (JSON path, Image Dir)
    pairs = [] # List of tuples (json_path, image_dir)
    
    # 1. Normal folders
    ann_dir = os.path.join(args.input, "annotations")
    img_dir = os.path.join(args.input, "images")
    if os.path.exists(ann_dir) and os.path.exists(img_dir):
        for f in os.listdir(ann_dir):
            if f.endswith(".json"):
                pairs.append((os.path.join(ann_dir, f), img_dir))

    # 2. Healthy folders
    h_ann_dir = os.path.join(args.input, "healthy_annotations")
    h_img_dir = os.path.join(args.input, "healthy_images")
    if os.path.exists(h_ann_dir) and os.path.exists(h_img_dir):
        for f in os.listdir(h_ann_dir):
            if f.endswith(".json"):
                pairs.append((os.path.join(h_ann_dir, f), h_img_dir))

    print(f"Found {len(pairs)} JSON files total.")

    # Process all files
    all_data = [] # Stores (image_info, annotations, src_img_path)
    global_img_id = 0
    
    for json_path, img_source_dir in pairs:
        img_info, anns, success, src_path = process_single_pair(json_path, img_source_dir, global_img_id, str_to_id)
        if success:
            all_data.append({
                "image": img_info,
                "annotations": anns,
                "src_path": src_path
            })
            global_img_id += 1

    # Shuffle
    random.seed(42)
    random.shuffle(all_data)

    # Calculate splits
    n_total = len(all_data)
    n_train = int(n_total * split_ratios[0])
    n_valid = int(n_total * split_ratios[1])
    # n_test is remainder

    train_data = all_data[:n_train]
    valid_data = all_data[n_train:n_train+n_valid]
    test_data = all_data[n_train+n_valid:]

    sets = [
        ("train", train_data),
        ("valid", valid_data),
        ("test", test_data)
    ]

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    info_dict = {
        "year": int(current_date.split("-")[0]),
        "version": "1.0",
        "description": "Converted from custom format",
        "contributor": "",
        "url": "",
        "date_created": current_date
    }
    
    licenses = [{
        "id": 1,
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/"
    }]

    global_ann_id = 0

    print(f"Processing split: Train({len(train_data)}), Valid({len(valid_data)}), Test({len(test_data)})")

    for set_name, dataset in sets:
        if not dataset and split_ratios[["train", "valid", "test"].index(set_name)] == 0:
            continue

        # Create dirs
        out_set_dir = os.path.join(args.output, set_name)
        os.makedirs(out_set_dir, exist_ok=True)
        
        # Prepare JSON structure
        coco_output = {
            "info": info_dict,
            "licenses": licenses,
            "images": [],
            "annotations": [],
            "categories": categories
        }

        for item in dataset:
            img = item["image"]
            anns = item["annotations"]
            src = item["src_path"]

            # Copy image
            dst = os.path.join(out_set_dir, os.path.basename(src))
            shutil.copy2(src, dst)

            coco_output["images"].append(img)
            
            # Re-id annotations specifically for this export (though global ID ensures uniqueness across splits too)
            for ann in anns:
                ann["id"] = global_ann_id
                coco_output["annotations"].append(ann)
                global_ann_id += 1

        # Write JSON
        json_out_path = os.path.join(out_set_dir, "_annotations.coco.json")
        with open(json_out_path, "w", encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False, indent=2)
        
        print(f"Created {set_name} set with {len(coco_output['images'])} images at {json_out_path}")

if __name__ == "__main__":
    main()