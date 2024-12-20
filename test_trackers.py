from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os
import argparse
import motmetrics as mm
import pandas as pd
import traceback
from scipy.optimize import linear_sum_assignment
from yolox.tracker.byte_tracker import BYTETracker
import time
from tqdm import tqdm

# Define classes we want to detect
CLASSES_OF_INTEREST = {
   0: 'person',
   2: 'car', 
   3: 'motorcycle',
   5: 'bus',
   7: 'truck',
}

Gt_Object_Classes = {
    0: 'Pedestrian',
    1: 'Cyclist',
    2: 'Motorbike',
    3: 'Small_motorised_vehicle',
    4: 'Car',
    5: 'Medium_vehicle',
    6: 'Large_vehicle',
    7: 'Bus',
    8: 'Emergency_vehicle'
}

class DetectionSource:
    YOLO = "yolo"
    GT = "gt"

class TrackerConfig:
    """Configuration for different trackers"""
    def __init__(self, 
                 track_thresh=0.25,
                 track_buffer=30,
                 match_thresh=0.8,
                 aspect_ratio_thresh=3.0,
                 mot20=False,
                 track_high_thresh=0.6,    # For BOT/BOOST
                 track_low_thresh=0.1,     # For BOT/BOOST
                 new_track_thresh=0.7,     # For BOT/BOOST
                 fps=10):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.mot20 = mot20
        # Additional parameters for other trackers
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.fps = fps

    @classmethod
    def byte_tracker_config(cls):
        """Default configuration for ByteTracker"""
        return cls(
            track_thresh=0.5,
            track_buffer=10,
            match_thresh=0.8
        )

    @classmethod
    def bot_tracker_config(cls):
        """Default configuration for BOT-SORT"""
        return cls(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            track_high_thresh=0.6,
            track_low_thresh=0.1
        )

    @classmethod
    def boost_tracker_config(cls):
        """Default configuration for BoostTracker"""
        return cls(
            track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.7,
            new_track_thresh=0.7
        )

class Trackers:
    """Available tracker types"""
    # User input names
    byte_tracker = "byte"
    bot_tracker = "bot"
    boost_tracker = "boost"

    # Mapping of user inputs to folder names
    _folder_names = {
        byte_tracker: "ByteTracker",
        bot_tracker: "BOT-SORT",
        boost_tracker: "BoostTracker"
    }

    @staticmethod
    def get_config(tracker_name):
        """Get default configuration for specified tracker"""
        if tracker_name not in [Trackers.byte_tracker, Trackers.bot_tracker, Trackers.boost_tracker]:
            raise ValueError(f"Unknown tracker: {tracker_name}. Use 'byte', 'bot', or 'boost'")
            
        config_map = {
            Trackers.byte_tracker: TrackerConfig.byte_tracker_config(),
            Trackers.bot_tracker: TrackerConfig.bot_tracker_config(),
            Trackers.boost_tracker: TrackerConfig.boost_tracker_config()
        }
        return config_map[tracker_name]
    
    @staticmethod
    def get_folder_name(tracker_name):
        """Get folder name for the tracker"""
        if tracker_name not in Trackers._folder_names:
            raise ValueError(f"Unknown tracker: {tracker_name}. Use 'byte', 'bot', or 'boost'")
        return Trackers._folder_names[tracker_name]
def get_tracker(tracker_name, fps, config=None):
    """
    Initialize tracker with given configuration
    Args:
        tracker_name: Type of tracker to use
        fps: Video FPS
        config: Optional custom configuration, if None uses default
    """
    if config is None:
        config = Trackers.get_config(tracker_name)
    config.fps = fps  # Update FPS in config

    if tracker_name == Trackers.byte_tracker:
        return BYTETracker(config, fps)
    elif tracker_name == Trackers.bot_tracker:
        raise NotImplementedError("BOT-SORT tracker not implemented")
    elif tracker_name == Trackers.boost_tracker:
        raise NotImplementedError("BOOST tracker not implemented")
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")
def load_gt_detections(gt_file):
    """
    Load ground truth detections in KITTI format.
    KITTI format: frame_id track_id object_class truncation occlusion alpha 
                 bbox_left bbox_top bbox_right bbox_bottom height width length 
                 x y z rotation_y score
    Returns: dict with frame_id as key, values as [x1,y1,x2,y2,score,class_id,track_id]
    """
    print("Loading GT file:", gt_file)
    gt_detections = {}
    
    # Create reverse mapping from class names to IDs
    class_name_to_id = {name.lower(): id for id, name in Gt_Object_Classes.items()}
    
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 17:  # KITTI format has 17 fields
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    class_name = parts[2]
                    
                    # Get class ID from our mapping
                    class_id = class_name_to_id.get(class_name.lower())
                    if class_id is None:
                        continue  # Skip if class not in our mapping
                    
                    # Get bbox coordinates
                    x1 = float(parts[6])
                    y1 = float(parts[7])
                    x2 = float(parts[8])
                    y2 = float(parts[9])
                    score = float(parts[-1]) if len(parts) > 17 else 1.0
                    
                    # Initialize frame list if needed
                    if frame_id not in gt_detections:
                        gt_detections[frame_id] = []
                    
                    # Store in our format: [x1,y1,x2,y2,score,class_id,track_id]
                    gt_detections[frame_id].append([x1, y1, x2, y2, score, class_id, track_id])
                    
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    print(f"Loaded {len(gt_detections)} frames of ground truth data")
            
    return gt_detections

def get_detections(frame, frame_counter, detection_type, model=None, gt_detections=None):
    """Get detections from specified source"""
    if detection_type == DetectionSource.YOLO:
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        mask = np.array([int(cls) in CLASSES_OF_INTEREST.keys() for cls in boxes.cls.cpu()])
        
        if len(boxes) > 0 and mask.any():
            filtered_boxes = boxes.xyxy[mask].cpu().numpy()
            filtered_scores = boxes.conf[mask].cpu().numpy()
            filtered_classes = boxes.cls[mask].cpu().numpy().astype(int)
            dets = np.concatenate([filtered_boxes, filtered_scores[:, None]], axis=1)
            return dets, filtered_classes, None
            
    elif detection_type == DetectionSource.GT:
        # print(f"gt_detections :{gt_detections}")
        if frame_counter in gt_detections:
            frame_dets = np.array(gt_detections[frame_counter])
            dets = frame_dets[:, :5]  # get x1,y1,x2,y2,score
            class_ids = frame_dets[:, 5].astype(int)  # get class_id
            track_ids = frame_dets[:, 6].astype(int)  # get track_id
            return dets, class_ids, track_ids

    return None, None, None

def save_to_kitti_format(detections, frame_id, output_file, frame_offset=0):
    """Save detections in KITTI format efficiently
    KITTI format per line:
    frame_id track_id object_class truncation occlusion alpha 
    bbox_left bbox_top bbox_right bbox_bottom height width length 
    x y z rotation_y score
    """
    # Prepare default values for 3D fields
    defaults = {
        'truncation': -1,
        'occlusion': -1,
        'alpha': -1,
        'height': -1,
        'width': -1,
        'length': -1,
        'x': -1,
        'y': -1,
        'z': -1,
        'rotation_y': -1
    }
    
    # Build all lines at once
    lines = []
    for box, track_id, class_id, conf in zip(detections.xyxy, 
                                           detections.tracker_id, 
                                           detections.class_id, 
                                           detections.confidence):
        if class_id not in Gt_Object_Classes:
            continue
            
        class_name = Gt_Object_Classes[class_id]
        x1, y1, x2, y2 = box
        
        # Format line with all fields
        line = f"{frame_id} {track_id} {class_name} "
        line += f"{defaults['truncation']} {defaults['occlusion']} {defaults['alpha']} "
        line += f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} "
        line += f"{defaults['height']:.2f} {defaults['width']:.2f} {defaults['length']:.2f} "
        line += f"{defaults['x']:.2f} {defaults['y']:.2f} {defaults['z']:.2f} "
        line += f"{defaults['rotation_y']:.2f} {conf:.2f}\n"
        
        lines.append(line)
    
    # Write all lines at once
    if lines:
        output_file.writelines(lines)

def process_batch(frames_batch, frame_indices, detection_type, model, gt_detections, tracker, 
                 height, width, use_gt_tracks, results_file, frame_offset):
    """Process a batch of frames and return detections and tracking data"""
    all_detections = []
    batch_tracking_data = []  # Store tracking data for batch writing
    
    for frame, frame_idx in zip(frames_batch, frame_indices):
        dets, class_ids, gt_track_ids = get_detections(frame, frame_idx, detection_type, model, gt_detections)
        
        if dets is not None:
            if use_gt_tracks:
                # Use ground truth tracks directly
                detections = sv.Detections(
                    xyxy=dets[:,:4],
                    confidence=np.ones(len(dets)),
                    class_id=class_ids,
                    tracker_id=gt_track_ids
                )
                batch_tracking_data.append((frame_idx, detections))
                all_detections.append((frame, detections))
            else:
                # Use tracker
                online_targets = tracker.update(dets, [height, width], [height, width])
                
                if len(online_targets) > 0:
                    track_boxes = np.array([t.tlwh for t in online_targets])
                    track_ids = np.array([t.track_id for t in online_targets])
                    track_scores = np.array([t.score for t in online_targets])
                    
                    track_boxes_xyxy = np.column_stack([
                        track_boxes[:, 0],
                        track_boxes[:, 1],
                        track_boxes[:, 0] + track_boxes[:, 2],
                        track_boxes[:, 1] + track_boxes[:, 3]
                    ])

                    detections = sv.Detections(
                        xyxy=track_boxes_xyxy,
                        confidence=track_scores,
                        class_id=class_ids[:len(track_boxes)],
                        tracker_id=track_ids
                    )
                    batch_tracking_data.append((frame_idx, detections))
                    all_detections.append((frame, detections))
                else:
                    all_detections.append((frame, None))
        else:
            all_detections.append((frame, None))
    
    # Save all detections from batch at once
    for frame_idx, detections in batch_tracking_data:
        save_to_kitti_format(detections, frame_idx, results_file, frame_offset)
    
    return all_detections
def load_frame_batch(frame_counter, batch_size, total_frames, frames, frame_folder, cap, frame_files=None):
    """Load a batch of frames"""
    frames_batch = []
    frame_indices = []
    for i in range(batch_size):
        if frame_counter + i > total_frames:
            break
            
        if frames:
            frame = cv2.imread(os.path.join(frame_folder, frame_files[frame_counter + i - 1]))
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        if frame is not None:
            frames_batch.append(frame)
            frame_indices.append(frame_counter + i)
            
    return frames_batch, frame_indices

def handle_visualization(frame, detections, detection_type, box_annotator, label_annotator, writer, show_display):
    """Handle frame visualization and video writing with optimized performance"""
    if not (writer is not None or show_display):
        return False  # Skip if no visualization needed
    
    # Only copy frame once if needed
    annotated_frame = frame.copy() if (detections is not None and (writer is not None or show_display)) else frame
    
    if detections is not None:
        # Prepare labels only if needed
        if show_display or writer is not None:
            if detection_type == DetectionSource.GT:
                labels = [f"#{t_id} {Gt_Object_Classes[c_id]}"
                         for t_id, c_id in zip(detections.tracker_id, detections.class_id)]
            else:
                labels = [f"#{t_id} {CLASSES_OF_INTEREST[c_id]}"
                         for t_id, c_id in zip(detections.tracker_id, detections.class_id)]

            # Annotate frame in-place
            box_annotator.annotate(annotated_frame, detections=detections)
            label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    # Handle display and video writing
    if writer is not None:
        writer.write(annotated_frame)
        
    if show_display:
        # Resize for display if frame is too large
        display_frame = annotated_frame
        if annotated_frame.shape[1] > 1280:  # If width > 1280
            scale = 1280 / annotated_frame.shape[1]
            display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
        
        cv2.imshow('Detections', display_frame)
        key = cv2.waitKey(1)
        return key & 0xFF == ord('q')
    
    return False

def process_detection(video_path, output_path, frames, frame_folder, frame_offset,
                     gt_file, detection_type,use_gt_tracks, show_display, 
                     save_video, batch_size, results_path,tracker_name="byte"):
    """Process video/frames and save tracking results"""
    video_name = os.path.basename(frame_folder.rstrip('/')) if frames else os.path.splitext(os.path.basename(video_path))[0]
    
    # Initialize detector
    model = YOLO("yolo11x.pt") if detection_type == DetectionSource.YOLO else None
    gt_detections = load_gt_detections(gt_file) if detection_type == DetectionSource.GT else None
   

    # Get video properties
    if frames:
        frame_files = sorted(os.listdir(frame_folder))
        first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
        height, width = first_frame.shape[:2]
        total_frames = len(frame_files)
        fps = 10
    else:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize tracker components
    try:
        if use_gt_tracks:
            tracker = None
            print("\nUsing ground truth tracks - No Tracker Initialized")
        else:
            print(f"Initializing {Trackers.get_folder_name(tracker_name)} with {fps} FPS")
            try:
                tracker = get_tracker(tracker_name, fps)
                print(f"Successfully initialized {Trackers.get_folder_name(tracker_name)}")
            except NotImplementedError as e:
                print(f"Tracker not implemented: {Trackers.get_folder_name(tracker_name)}")
                raise e
            except Exception as e:
                print(f"Failed to initialize {Trackers.get_folder_name(tracker_name)}: {str(e)}")
                raise e

    except Exception as e:
        print(f"Error in tracker initialization: {e}")
        return

    
    box_annotator = sv.BoxAnnotator() if (show_display or save_video) else None
    label_annotator = sv.LabelAnnotator() if (show_display or save_video) else None
    
    # Initialize output
    results_file = open(results_path, 'w')
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if save_video else None
    
    # Process frames
    frame_counter = 1
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while frame_counter <= total_frames:
            
            frames_batch, frame_indices = load_frame_batch(
                frame_counter, batch_size, total_frames, frames, frame_folder, 
                None if frames else cap, frame_files if frames else None
            )
            
            if not frames_batch:
                break

            processed_frames = process_batch(
                frames_batch, frame_indices, detection_type, model, gt_detections,
                tracker, height, width, use_gt_tracks, results_file, frame_offset
            )

            if show_display or save_video:
                for frame, detections in processed_frames:
                    if handle_visualization(frame, detections, detection_type, 
                                         box_annotator, label_annotator, writer, show_display):
                        break

            pbar.update(len(frames_batch))
            frame_counter += len(frames_batch)

    # Cleanup
    results_file.close()
    if not frames:
        cap.release()
    if writer:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()

def process_video_folder(video_folder, output_folder, gt_folder=None, use_frames=False, 
                        detection_type=DetectionSource.YOLO,tracker_name='ByteTracker', use_gt_tracks=False, 
                        show_display=False, save_video=False, batch_size=32):
    
    """Process videos and save results in TrackEval format"""
    
    # Get tracker folder name
    if use_gt_tracks:
        tracker_folder = 'GtTracker'
    else:
        tracker_folder = Trackers.get_folder_name(tracker_name)
    
    # Create tracker-specific output folders
    pred_data_folder = os.path.join(output_folder, tracker_folder, "tracked_predictions")
    os.makedirs(pred_data_folder, exist_ok=True)
    
    track_video_folder = os.path.join(output_folder, tracker_folder, "tracked_videos")
    os.makedirs(track_video_folder, exist_ok=True)

    # Get list of input files
    all_items = os.listdir(video_folder)
    if use_frames:
        items = sorted([d for d in all_items if os.path.isdir(os.path.join(video_folder, d))])
    else:
        items = sorted([f for f in all_items if f.endswith(('.mp4', '.avi', '.mov'))])

    print(f"\nProcessing {len(items)} {'frame folders' if use_frames else 'videos'}")
    

    for item in items:
        item_name = item.split('.')[0]
        print(f"\nProcessing {item_name}")
        
        # Setup paths
        if use_frames:
            frame_folder = os.path.join(video_folder, item)
            video_path = None
            output_video = os.path.join(track_video_folder, f"{item_name}_tracked.mp4") if save_video else None
        else:
            video_path = os.path.join(video_folder, item)
            frame_folder = None
            output_video = os.path.join(track_video_folder, f"{item_name}_tracked.mp4") if save_video else None

        # Setup output file for tracking results
        results_path = os.path.join(pred_data_folder, f"{item_name}.txt")
        
        # Get GT file path from gt_folder
        gt_file = os.path.join(gt_folder, f"{item_name}.txt") if gt_folder else None
        
        if gt_file and not os.path.exists(gt_file):
            print(f"Warning: GT file not found for {item_name}")
        

        try:
            process_detection(
                video_path=video_path,
                output_path=output_video,
                frames=use_frames,
                frame_folder=frame_folder,
                frame_offset=0,
                gt_file=gt_file,
                detection_type=detection_type,
                use_gt_tracks=use_gt_tracks,
                show_display=show_display,
                save_video=save_video,
                batch_size=batch_size,
                results_path=results_path,
                tracker_name=tracker_name
            )
        except Exception as e:
            print(f"Error processing {item_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\nResults saved in TrackEval format to: {output_folder}")
    return output_folder


def print_args(args, verbose=False):
    """
    Print argument values in either verbose or simple format
    
    Args:
        args: Parsed argument namespace from argparse
        verbose: If True, print detailed output with descriptions
    """
    arg_dict = vars(args)
    
    if verbose:
        print("\n" + "="*50)
        print("CONFIGURATION SETTINGS")
        print("="*50)
        
        descriptions = {
            'input': 'Input folder path (videos/frames)',
            'output': 'Output folder path',
            'gt_folder': 'Ground truth annotations folder',
            'Tracker_name': 'Tracker to be used',
            'use_frames': 'Using frame folders as input',
            'det': 'Detection source (YOLO/GT)',
            'gt_gt': 'Using ground truth for detection and tracking',
            'show_display': 'Visualization enabled',
            'save_video': 'Output video saving enabled',
            'batch_size': 'Processing batch size',
            'verbose': 'Verbose output mode'
        }
        
        max_arg_len = max(len(arg) for arg in arg_dict.keys())
        max_val_len = max(len(str(val)) for val in arg_dict.values())
        
        for arg, value in arg_dict.items():
            description = descriptions.get(arg, 'No description available')
            print(f"{arg:<{max_arg_len}} : {str(value):<{max_val_len}} | {description}")
        
        print("="*50 + "\n")
    else:
        essential_args = ['input', 'output', 'det', 'batch_size']
        bool_flags = ['use_frames', 'gt_gt', 'show_display', 'save_video', 'verbose']
        
        parts = [f"{arg}={arg_dict[arg]}" for arg in essential_args]
        enabled_flags = [arg for arg in bool_flags if arg_dict[arg]]
        
        if enabled_flags:
            parts.extend(enabled_flags)
            
        print("\nConfig:", ", ".join(parts) + "\n")

if __name__ == "__main__":

    # Create parser with print_args functionality built in
    parser = argparse.ArgumentParser(description='Process videos with tracking')
    parser.add_argument('--input', type=str, default='emt/frames/', 
                    help='Path to input folder (videos or frame folders)')
    parser.add_argument('--output', type=str,  default="Trackers/", 
                    help='Path to output folder')
    parser.add_argument('--tracker_name', type=str,  default="byte", 
                    help='Tracker name use byte, bot or boost')
    parser.add_argument('--gt_folder', type=str, default='emt/emt_annotations/labels/',
                    help='Path to folder containing GT files')
    parser.add_argument('--use_frames', default=True,action='store_true', 
                    help='Input folder contains frame folders instead of videos')
    parser.add_argument('--det', type=str, choices=['yolo', 'gt'], default='gt', 
                    help='Detection source')
    parser.add_argument('--use_gt_tracks', action='store_true', 
                    help='Use ground truth detections and tracking')
    parser.add_argument('--show_display', action='store_true', 
                    help='Show visualization')
    parser.add_argument('--save_video', action='store_true', 
                    help='Save output videos')
    parser.add_argument('--batch_size', type=int, default=1, 
                    help='Batch size for processing')
    parser.add_argument('--verbose', default=True,action='store_true',
                    help='Enable verbose output mode')
    
    
    args = parser.parse_args()
    
    print_args(args, args.verbose)  # Print configuration settings
    
    print("\nStarting batch processing...")
    t0 = time.time()
    
    process_video_folder(
        video_folder=args.input,
        output_folder=args.output,
        gt_folder=args.gt_folder,
        use_frames=args.use_frames,
        detection_type=args.det,
        tracker_name = args.tracker_name,
        use_gt_tracks=args.use_gt_tracks,
        show_display=args.show_display,
        save_video=args.save_video,
        batch_size=args.batch_size
    )
    
    t1 = time.time()
    total_time = t1 - t0
    """Convert seconds to hours:minutes:seconds format"""
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    time_spent = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}" 
    
    print("\nBatch Processing Statistics:")
    print("-" * 50)
    print(f"Total processing time: {time_spent}")
    print("-" * 50)
