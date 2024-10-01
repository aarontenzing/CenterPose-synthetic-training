import json
import matplotlib.pyplot as plt
import os
import numpy as np
from src.lib.utils.pnp.cuboid_pnp_shell import pnp_shell
from src.lib.opts import opts
import sys
from src.tools.objectron_eval.objectron.dataset.box import Box as Boxcls
from src.tools.objectron_eval.objectron.dataset.iou import IoU
import pickle
from pprint import pprint
import cv2
from tqdm import tqdm


def get_gt_points(dict, meta, opt):
    size = dict["whd"] # object size
    points = dict["projection"][:8] # GT 2D points verices
   
    try: 
        # PnP 
        bbox= {'kps': points, "obj_scale": size} # normalize y
        projected_points, point_3d_cam, scale, points_ori, bbox = pnp_shell(opt, meta, bbox, points, size, OPENCV_RETURN=False)

    except:
        print("GT wrong point order")
        return [0]
    
    # return the 3D world points after PnP:
    return np.array(bbox["kps_3d_cam"])

def get_annotations():
    # Get annotations in test/anno.json
    with open(root_json_gt, "r") as f:
        data = json.load(f)

    return data

def translate_along_optical_axis(gt_box, detect_box):
    optical_center = np.array([0, 0, 0])  # Optical center at world origin (0,0,0)
    
    # Extract center points from both boxes (assuming first element is the center)
    center_gt = gt_box.vertices[0]  # Ground truth box center (first vertex)
    center_detect = detect_box.vertices[0]  # Detection box center (first vertex)

    # Get the optical axis (vector from optical center to the detection box center)
    optical_line = center_detect - optical_center
    optical_line = optical_line / np.linalg.norm(optical_line)  # Normalize the vector

    # Compute the translation required to move the detection center to the ground truth center
    translation_vector = center_gt - center_detect
    shift_magnitude = np.dot(translation_vector, optical_line)  # Project shift on the optical axis

    # Apply this translation to the detection center and all its vertices
    translated = []
    for vertex in detect_box.vertices:
        translated_vertex = vertex + optical_line * shift_magnitude  # Translate along optical axis
        translated.append(translated_vertex)

    # Convert the list of translated vertices to a numpy array (if required by your Boxcls)
    translated = np.array(translated)
    translated_box = Boxcls(translated)

    return translated_box

def evaluate_img(root_json_detect, img_id, verbose=False):
    img_list = [file for file in os.listdir(root_img) if file.endswith(".jpg")]
    img_name = img_id + ".jpg"

    if img_name not in img_list:
        print("Image not found or is not a box")
        return 1,0
        
    annotations = get_annotations()
    annotation = annotations[int(img_id)]

    img = plt.imread(root_img + img_name)

    if verbose: # uitgebreid
        img = plt.imread(root_img + img_name)
        plt.imshow(img)
        plt.show()

    # Finding the annotation found after inference:
    # Check if the image has a debug file:
    detections = os.listdir(root_json_detect)
    detect_json = img_id + ".json"

    if detect_json not in detections:
        print("no detection json found")
        return 1,0

    # open detection json:
    with open(root_json_detect + img_id + ".json", "r") as f:
        detection = json.load(f)
    
    # Check if the detection has no objects:
    if len(detection["objects"]) == 0:
        print("no detection")
        return 1, 0
    
    detection_points = np.array(detection["objects"][0]["kps_3d_cam"]) # detected 3d world coordinates
    
    # OPT:
    opt = opts()
    opt.nms = True
    opt.obj_scale = True
    opt.c = "cereal_box" # category
    
    # Meta: 
    # Load data from the PKL file
    with open('cameraMatrix.pkl', 'rb') as f:
        camera = pickle.load(f)

    meta = {"width": img.shape[1],"height": img.shape[0], "camera_matrix":camera }
    
    # Get 3D GT:
    print("img_id", img_id)
    gt_points = get_gt_points(annotation, meta, opt) # doet pnp op gt pixel coördinaten -> om de 3D wereld coordinaten te krijgen
    if len(gt_points) == 1:
        print("wrong annotation point order")
        return 1,0
    
    # Make box objects and determine IoU:
    gt_box = Boxcls(gt_points) 
    detect_box = Boxcls(detection_points)

    iou = IoU(detect_box, gt_box) # calculate IoU:
    result = iou.iou()
    print('Old IOU:', result)

    # Shift the box in the z-direction so it falls on middelpoint of detection:
    # z_trans = gt_box.vertices[0][2] - detect_box.vertices[0][2]
    # translated = []
    # for i in range(len(detect_box.vertices)):
    #         translated_vertex = detect_box.vertices[i].copy()  # Assuming vertices are lists or arrays
    #         translated_vertex[2] += z_trans  # Apply translation to the z component
    #         translated.append(translated_vertex)
    
    # translated = np.array(translated)
    # translated_box = Boxcls(translated)

    # # Shift the box so it falls on middelpoint of detection:
    # trans = gt_box.vertices[0] - detect_box.vertices[0]
    # translated = []
    # for i in range(len(detect_box.vertices)):
    #     translated.append(detect_box.vertices[i]+trans)
    
    # translated = np.array(translated)
    # translated_box = Boxcls(translated)

    # print("translated points: \n", translated)
    # print("volume detected", detect_box.volume)
    # print("volume GT",gt_box.volume)
    # print("scale detected", detect_box.scale)
    # print("scale GT",gt_box.scale)

    # Translate via optical axis
    translated_box = translate_along_optical_axis(gt_box, detect_box)
    
    iou = IoU(translated_box, gt_box)
    result = iou.iou()
    print("IOU: results", result)

    return result, img


def write_iou(id, iou, filepath):
    # Load JSON data from file
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Update JSON with IOU value
    data[id]["IOU"] = iou

    # Write updated JSON back to file
    with open(filepath, 'w') as file:
        json.dump(data, file, indent = 4)
        # print("write!")

  
def main(img_id):
    # Dir with detections:
    img_id = str(img_id) # (0, 1, 2, 3...)
    iou, img = evaluate_img(root_json_detect, img_id)

    failed = 0
    if iou == 1:
        print("failed to detect")
        write_iou(int(img_id), "None", root_json_gt)
    else:
        write_iou(int(img_id), iou, root_json_gt)
        iou *= 100
        # print(f"the intersection of union was {round(iou)}, see the plot for the result")
        


def find_iou(root_json_detect, file):
    with open(root_json_detect + file, "r") as f:
        data = json.load(f)

    if len(data["objects"]) == 0:
        print("No object found!")
        return 0

    return data["objects"][0]["IOU"]

     
def get_statistics(dection_results, verbose=False):

    # TEST ANNOTATIONS + IOU:
    with open(dection_results, "r") as f:
        data = json.load(f)
    
    total_test = len(data)
    print("Total amount of images: ", total_test)

    missed_test = 0
    found_test = 0

    failed_test = 0
    correct_test_25 = 0
    correct_test_50 = 0
    correct_test_75 = 0
    total_iou = 0
    
    # Go through detection results:
    for info in data:
        name = int(info["image"])
        iou = info["IOU"]
        if iou == "None":
            found = False
            missed_test += 1
        else:
            found = True
            iou = float(iou)
            found_test += 1

        # Find IoU and add statistics   
        if found:
            total_iou += iou
            if iou >= 0.25:
                correct_test_25 += 1
                if iou >= 0.5:
                    correct_test_50 += 1
                    if iou >= 0.75:
                        correct_test_75 += 1
            else:
                failed_test += 1

    print()
    print("TEST: ") 
    print(f"The total number of samples is {total_test}")
    print(f"The total number of missed is {missed_test}")
    print(f"The total average IOU is {format(total_iou / (total_test), '.4f')}")
    print(f"The IOU > 25% {format(correct_test_25 / (total_test), '.4f')}")
    print(f"The IOU > 50% {format(correct_test_50 / (total_test), '.4f')}")
    print(f"The IOU > 75% {format(correct_test_75 / (total_test), '.4f')}")
    print(f"The % found boxes is {format((found_test) / total_test, '.4f')}")
    print(f"The % missed boxes is {format((missed_test) / total_test, '.4f')}")


test_type = "real_test"
experiment_folder = "real_test"
# test_type = "synthetic_test"

root_img = "data/synthetic_data/" + test_type + "/" # CHANGE THIS
root_json_detect = "exp/" + experiment_folder + "/" # CHANGE THIS
root_json_gt = root_img + "anno.json"


if __name__ == "__main__":
    test_images = os.listdir(root_img) 
    test_images = [file for file in test_images if file.endswith(".jpg")]
    test_images = sorted(test_images, key=lambda x: int(x.split(".")[0]))

    with tqdm(total=len(test_images), desc="Calculating IoUs") as pbar:
        # Go through images and calculate IOU and write
        for img_id in test_images:
            main(img_id.split('.')[0])
            pbar.update(1)

    get_statistics(root_json_gt)