# Copyright (c) Tencent Inc. All rights reserved.
# This file is modified from mmyolo/demo/video_demo.py
import argparse
import os

import cv2
import mmcv
import torch
import numpy as np
import supervision as sv
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmengine.utils import ProgressBar

class CustomLabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World video demo')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('video', help='video file path')
    parser.add_argument(
        'text',
        help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference')
    parser.add_argument('--score-thr',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--out', type=str, help='output video file')
    parser.add_argument('--label-size', type=float, default=0.5, help='label font size')
    parser.add_argument('--box-thickness', type=int, default=2, help='bounding box thickness')
    args = parser.parse_args()
    return args

def inference_detector(model, image, texts, test_pipeline, score_thr=0.3):
    data_info = dict(img_id=0, img=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    output.pred_instances = pred_instances
    return output

def main():
    args = parse_args()
    
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
        custom_classes = [t[0] for t in texts[:-1]]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]
        custom_classes = [t[0] for t in texts[:-1]]

    model.reparameterize(texts)

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=args.box_thickness,
        color=sv.Color(r=0, g=255, b=0)
    )
    
    label_annotator = CustomLabelAnnotator(
        text_padding=4,
        text_scale=args.label_size,
        text_thickness=1,
        text_color=sv.Color(r=255, g=255, b=255),
        text_position=sv.Position.TOP_LEFT
    )

    video_reader = mmcv.VideoReader(args.video)
    total_frames = len(video_reader)

    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    progress_bar = ProgressBar(total_frames)
    
    for i in range(total_frames):
        frame = video_reader[i]
        progress_bar.update()
        
        result = inference_detector(model,
                                    frame,
                                    texts,
                                    test_pipeline,
                                    score_thr=args.score_thr)
        
        pred_instances = result.pred_instances.cpu().numpy()
        
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores']
        )
        
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            if 0 <= class_id < len(custom_classes):
                label = f"{custom_classes[class_id]} {confidence:.2f}"
            else:
                label = f"unknown {confidence:.2f}"
            labels.append(label)
        
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)

        if video_writer:
            video_writer.write(annotated_frame)
    
    if video_writer:
        video_writer.release()
    
    print(f"\nProcessing complete! Output saved to: {args.out}")

if __name__ == '__main__':
    main()