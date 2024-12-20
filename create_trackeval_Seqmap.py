import os
# Directories for ground truth and evaluation
gt_fol = 'emt/emt_annotations_trackeval/label_02/'
evaluate_tracking_fol = 'emt/emt_annotations_trackeval/'

# Get a list of all ground truth files in the folder
gt_files = [f for f in os.listdir(gt_fol) if f.endswith('.txt')]

# Extract sequence IDs (assuming the format 'video_<ID>.txt')
seq_ids = [f.split('_')[1].split('.')[0] for f in gt_files]  # Extract the numeric part

# Path to save the seqmap file
seqmap_path = os.path.join(evaluate_tracking_fol, 'evaluate_tracking.seqmap.val')

# Create seqmap file
with open(seqmap_path, 'w') as f:
    for seq_id in seq_ids:
        # Ensure the sequence ID is valid (non-empty)
        if seq_id:
            # Count the number of unique frames for this sequence
            curr_file = os.path.join(gt_fol, f'video_{seq_id}.txt')
            if os.path.isfile(curr_file):
                unique_frames = set()  # Use a set to store unique frame IDs
                
                with open(curr_file, 'r') as gt_fp:
                    for line in gt_fp:
                        parts = line.strip().split()  # Split the line into parts
                        if parts:
                            frame_id = parts[0]  # Assuming the frame ID is the first part
                            unique_frames.add(frame_id)  # Add frame ID to the set

                num_frames = len(unique_frames)  # The length is the number of unique frame IDs

                # Write the sequence ID and its length (number of unique frames) to the seqmap file
                f.write(f"{seq_id} {num_frames}\n")
            else:
                print(f"Ground truth file for sequence {seq_id} not found.")
                # Optionally raise an exception or handle the error differently

print(f"Seqmap file created at: {seqmap_path}")
