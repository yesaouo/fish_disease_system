"""VOC2007+2012 -> COCO for RF-DETR training.
train = VOC2007 trainval + VOC2012 trainval ; valid = VOC2007 test.
difficult objects dropped. images symlinked into split dirs.
"""
import os, json, xml.etree.ElementTree as ET

VOCROOT = "diagnosis_model/vl_classifier/voc_pipeline/data/VOCdevkit"
OUT = "data/detection_voc"
CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
           "cow","diningtable","dog","horse","motorbike","person","pottedplant",
           "sheep","sofa","train","tvmonitor"]
CAT_ID = {c: i+1 for i, c in enumerate(CLASSES)}
SPLITS = {"train": [("VOC2007","trainval"), ("VOC2012","trainval")],
          "valid": [("VOC2007","test")]}

def ids(voc, split):
    with open(f"{VOCROOT}/{voc}/ImageSets/Main/{split}.txt") as f:
        return [l.strip() for l in f if l.strip()]

def build(split, specs):
    outdir = f"{OUT}/{split}"; os.makedirs(outdir, exist_ok=True)
    images, annos = [], []
    img_id, ann_id = 0, 0
    for voc, sub in specs:
        for name in ids(voc, sub):
            jpg = f"{VOCROOT}/{voc}/JPEGImages/{name}.jpg"
            if not os.path.exists(jpg) or os.path.getsize(jpg) == 0:
                print(f"  skip corrupt/empty image: {voc}/{name}.jpg")
                continue
            xml = f"{VOCROOT}/{voc}/Annotations/{name}.xml"
            root = ET.parse(xml).getroot()
            size = root.find("size")
            W, H = int(size.find("width").text), int(size.find("height").text)
            fn = f"{voc}_{name}.jpg"
            link = f"{outdir}/{fn}"
            if not os.path.islink(link) and not os.path.exists(link):
                os.symlink(os.path.abspath(jpg), link)
            img_id += 1
            images.append({"id": img_id, "file_name": fn, "width": W, "height": H})
            for obj in root.findall("object"):
                if obj.find("difficult") is not None and obj.find("difficult").text == "1":
                    continue
                cls = obj.find("name").text
                if cls not in CAT_ID:
                    continue
                b = obj.find("bndbox")
                x1, y1 = float(b.find("xmin").text)-1, float(b.find("ymin").text)-1
                x2, y2 = float(b.find("xmax").text)-1, float(b.find("ymax").text)-1
                x1, y1 = max(0, x1), max(0, y1)
                w, h = x2-x1, y2-y1
                if w <= 0 or h <= 0:
                    continue
                ann_id += 1
                annos.append({"id": ann_id, "image_id": img_id,
                              "category_id": CAT_ID[cls], "bbox": [x1, y1, w, h],
                              "area": w*h, "iscrowd": 0, "segmentation": []})
    coco = {"images": images, "annotations": annos,
            "categories": [{"id": CAT_ID[c], "name": c, "supercategory": "object"} for c in CLASSES]}
    with open(f"{outdir}/_annotations.coco.json", "w") as f:
        json.dump(coco, f)
    print(f"{split}: {len(images)} imgs, {len(annos)} boxes -> {outdir}")

if __name__ == "__main__":
    for s, sp in SPLITS.items():
        build(s, sp)
