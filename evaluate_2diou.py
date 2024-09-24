import json
import os
import numpy as np
from shapely.geometry import box
import cv2

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (tuple): Bounding box in format (x1, y1, x2, y2).
        box2 (tuple): Bounding box in format (x1, y1, x2, y2).
        
    Returns:
        float: IoU value between 0 and 1.
    """
    # Convert boxes to Shapely box objects
    rect1 = box(*box1)
    rect2 = box(*box2)
    
    # Calculate the intersection area
    intersection = rect1.intersection(rect2).area
    
    # Calculate the union area
    union = rect1.union(rect2).area
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    # Calculate IoU
    return intersection / union

def get_ground_truth_annotations(filepath):
    """
    Load ground truth annotations from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file containing ground truth annotations.
        
    Returns:
        dict: Ground truth annotations.
    """
    with open(filepath, "r") as f:
        return json.load(f)

def get_detection_results(filepath):
    """
    Load detection results from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file containing detection results.
        
    Returns:
        dict: Detection results.
    """
    with open(filepath, "r") as f:
        return json.load(f)

def get_2d_bounding_box(points):
    points = np.array(points)
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    return [x_min, y_min, x_max, y_max]

def get_convexhull(corner_points):
    mask = np.zeros(shape, dtype=np.uint8)
    # Calculate the convex hull of the points to ensure proper ordering
    hull = cv2.convexHull(np.int32(corner_points))
    # Fill the polygon defined by the convex hull
    cv2.fillPoly(mask, [hull], 1)

    return mask

def calculate_iou_convex(mask1, mask2):
    # Calculate the intersection
    intersection = np.logical_and(mask1, mask2).sum()
    
    # Calculate the union
    union = np.logical_or(mask1, mask2).sum()
    
    # Compute IoU
    iou = intersection / union if union > 0 else 0
    return iou


def get_sorted_json_files(directory):
    """
    Get all JSON files in the directory, sorted by their numeric ID.
    
    Args:
        directory (str): Path to the directory containing the JSON files.
        
    Returns:
        list: List of sorted JSON filenames.
    """
    # Filter and sort JSON files based on their numeric ID
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    # Sort based on the numeric part of the file name (before the .json extension)
    sorted_files = sorted(json_files, key=lambda x: int(x.split('.')[0]))

    return sorted_files

def evaluate_iou(ground_truth_file, detection_file, output_file, img_id):
    """
    Evaluate 2D IoU for detections compared to ground truth.
    
    Args:
        ground_truth_file (str): Path to the JSON file with ground truth annotations.
        detection_file (str): Path to the JSON file with detection results.
        output_file (str): Path to save the evaluation results.
    """
    # Load data
    ground_truths = get_ground_truth_annotations(ground_truth_file) # GT
    detections = get_detection_results(detection_file) # DETECTION

    gt_data = ground_truths[img_id]
    gt_box = get_2d_bounding_box(gt_data["projection"]) # calculate gt bounding box

    try:
        det_box = detections["objects"][0].get("bbox")
        det_points = detections["objects"][0].get("kps_displacement_mean")
        x_coords = det_points[::2]  # Every second value starting from index 0 (x-coordinates)
        y_coords = det_points[1::2]  # Every second value starting from index 1 (y-coordinates)
        det_points = np.column_stack((x_coords, y_coords)).astype(int)


    except (IndexError, KeyError) as e:
        return 0, 0
        
    iou_bbox = calculate_iou(gt_box, det_box)
    # sythetic 512 x 512, real 3888 x 5184 
    mask1 = get_convexhull(corner_points=gt_data["projection"])
    mask2 = get_convexhull(corner_points=det_points)  

    # # # Convert masks to BGR format for visualization
    # mask1_colored = cv2.cvtColor(mask1 * 255, cv2.COLOR_GRAY2BGR)
    # mask2_colored = cv2.cvtColor(mask2 * 255, cv2.COLOR_GRAY2BGR)

    # # Resize with a scaling factor of 5
    # scale_factor = 1/5
    # mask1_resized = cv2.resize(mask1_colored, (0, 0), fx=scale_factor, fy=scale_factor)
    # mask2_resized = cv2.resize(mask2_colored, (0, 0), fx=scale_factor, fy=scale_factor)

    # # Show the masks
    # cv2.imshow("Ground Truth Mask", mask1_resized)
    # cv2.imshow("Detection Mask", mask2_resized)

    # key = cv2.waitKey(0)  # Wait for a key press
    # if key == 27:  # 27 is the ASCII code for the Esc key
    #     cv2.destroyAllWindows()
               
    iou_convex = calculate_iou_convex(mask1, mask2)

    return iou_bbox, iou_convex

if __name__ == "__main__":
    # Define paths
    # ALways put one in comment of the test types
    test_type = "real_test"
    # test_type = "synthetic_test"

    root_gt = "data/synthetic_data/" + test_type + "/anno.json"
    root_det = "exp/" + test_type + "/"
    image = cv2.imread("data/synthetic_data/" + test_type + "/0.jpg")

    shape = image.shape[:2]
    print(shape)

    output_file = "results_eval/2d_iou_results.json"
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    iou_results_bbox = []
    iou_results_convex = []

    sorted_json_files = get_sorted_json_files(root_det)

    # Evaluate IoU for each image
    for img_file in sorted_json_files:
        if img_file.endswith(".json"):
            img_id = img_file.split('.')[0] # takes the id of the image
            iou_bbox, iou_convex = evaluate_iou(root_gt, os.path.join(root_det, img_file), output_file, int(img_id))
            iou_results_bbox.append(iou_bbox)
            iou_results_convex.append(iou_convex)
    

    iou_results_bbox = np.array(iou_results_bbox)
    iou_results_convex = np.array(iou_results_convex)

    # Print results with a clearer format
    print("IOU Results BBox:")
    print("Shape:", iou_results_bbox.shape)
    print("Values:\n", np.array2string(iou_results_bbox, formatter={'float_kind': lambda x: f"{x:.4f}"}, threshold=10))
    print("Mean 2D IOU BBOX:", np.mean(iou_results_bbox))

    print("\nIOU Results Convex:")
    print("Shape:", iou_results_convex.shape)
    print("Values:\n", np.array2string(iou_results_convex, formatter={'float_kind': lambda x: f"{x:.4f}"}, threshold=10))
    print("Mean 2D IOU CONVEX:", np.mean(iou_results_convex))

    # non_zero_bbox = iou_results_bbox[iou_results_bbox > 0]
    # print("Non-Zero IOU Results BBox:")
    # print(non_zero_bbox)

    # non_zero_convex = iou_results_convex[iou_results_convex > 0]
    # print("\nNon-Zero IOU Results Convex:")
    # print(non_zero_convex)
