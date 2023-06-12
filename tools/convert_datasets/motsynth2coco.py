import argparse
from path import Path
import json
from collections import defaultdict
import os.path as osp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract MOT annotations from MOTSynth dataset.')
    parser.add_argument('--anns', help='path of MOTSynth annotations')
    parser.add_argument('--out', help='path to save MOT annotations')
    return parser.parse_args()


""" json file format:
{
    "categories": []
    "videos": [
        {
            "id": 0,
            "name": "000",
            "fps": 20,
            "width": 1920,
            "height": 1080,
        },
    ]
    "images": [
        {
            "file_name": "000/img1/000001.jpg",
            "height": 600,
            "width": 800,
            "id": 1 (global)
            "frame_id": 0 (local),
            "video_id": 0,
        }
    ]
    "annotations": [
        {
            "category_id": 1,
            "bbox": [
                x1,
                y1,
                w,
                h
            ],
            "area": w*h,
            "iscrowd": false,
            "id": 671566,
            "image_id": 49527,
            "instance_id": 1000
        }
    ]
    "num_instances": 100
}
"""


def main(args):
    input_path = Path(args.anns)
    outputs = defaultdict(list)
    outputs['categories'] = [dict(id=1, name='pedestrian')]
    videos = input_path.files()
    videos.sort(key=lambda x: int(x.basename().split('.')[0]))

    vid_id, img_id, ann_id = 1, 1, 1
    for ann in tqdm(videos):
        image_id_local2global = {}
        with open(ann, 'r') as f:
            data = json.load(f)

        # 1. process videos
        vid_name = ann.basename().split('.')[0]
        video = dict(
            id=vid_id,
            name=vid_name,
            fps=20,
            height=1080,
            width=1920,
        )
        outputs['videos'].append(video)

        # 2. process images
        for img_item in data['images']:
            img_name = osp.join(vid_name, 'img1', f'{img_item["frame_n"]:06}.jpg')
            frame_id = img_item['frame_n'] - 1
            image_id_local2global[img_item['id']] = img_id
            image = dict(
                file_name=img_name,
                height=img_item['height'],
                width=img_item['width'],
                id=img_id,
                frame_id=frame_id,
                video_id=vid_id,
            )
            outputs['images'].append(image)
            img_id += 1

        # 3. process annotations
        for ann_item in data['annotations']:
            ann = dict(
                category_id=1,
                bbox=ann_item['bbox'],
                area=ann_item['bbox'][2] * ann_item['bbox'][3],
                iscrowd=bool(ann_item['iscrowd']),
                id=ann_id,
                image_id=image_id_local2global[ann_item['image_id']],
                instance_id=ann_item['ped_id'],
            )
            outputs['annotations'].append(ann)
            ann_id += 1

        vid_id += 1
    outputs['num_instances'] = ann_id

    with open(args.out, 'w') as f:
        json.dump(outputs, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
