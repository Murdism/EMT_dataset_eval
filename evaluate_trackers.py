import os
import trackeval
# from trackeval import Evaluator, datasets, metrics

# Paths
EMT_GT_PATH = 'emt/emt_annotations/'  # Ground truth folder
EMT_TRACKERS_PATH = 'Trackers/'  # Tracker results folder

# Tracker and dataset details
TRACKER_NAME = 'GtTracker' #'ground_truth'  # Name of your tracker
TRACKER_SUB_FOLDER = 'tracked_predictions'#'gt'  # Leave empty if results are in the default folder

# Configurations
def get_emt_config():
    """Configuration for EMT evaluation."""
    return {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 20,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_CONFIG': True,
        'LOG_ON_ERROR': os.path.join(os.getcwd(), 'error_log.txt'),
        'GT_FOLDER': EMT_GT_PATH,
        'TRACKERS_FOLDER': EMT_TRACKERS_PATH,
        'TRACKER_SUB_FOLDER': TRACKER_SUB_FOLDER,
        'OUTPUT_FOLDER': os.path.join(os.getcwd(), 'EVAL_RESULTS'),
        'TRACKERS_TO_EVAL': [TRACKER_NAME],
        'CLASSES_TO_EVAL':['Pedestrian', 'Cyclist', 'Motorbike',  
                    'Car'],#,'Small_motorised_vehicle','Medium_vehicle', 'Large_vehicle', 'Bus', 'Emergency_vehicle',#  ['car', 'pedestrian'],#  # Select classes to evaluate
        'USE_SUPER_CATEGORIES': True,
        'BENCHMARK': 'kitti',  # KITTI benchmark
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],  # Metrics to evaluate
        'TRACKER_DISPLAY_NAMES': [TRACKER_NAME]
    }

# Main function to evaluate
if __name__ == '__main__':
    config = get_emt_config()
    dataset_list = [trackeval.datasets.EMT2DBox(config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    
    evaluator = trackeval.Evaluator(config)
    evaluator.evaluate(dataset_list, metrics_list)
