import json
import matplotlib.pyplot as plt
import os
import numpy as np
from src.lib.utils.pnp.cuboid_pnp_shell import pnp_shell
from src.lib.opts import opts
import sys
from src.tools.objectron_eval.objectron.dataset.box import Box as Boxcls
from src.tools.objectron_eval.objectron.dataset.iou import IoU

root_img="data/synthetic_data/val/"
root_json_gt="data/synthetic_data/anno.json"

def get_gt_points(dict, meta, opt):
    invert_points=[]
    for i, p in enumerate(dict["points"]):
        # invert_points.append([p[0], meta["height"]-p[1]])
        invert_points.append([p[0], meta["height"]])
        
    size = np.array(dict["size"]) # object size
   
    try: 
        # PnP 
        bbox= {'kps': invert_points[:8], "obj_scale": size/size[1]} # normalize y
        projected_points, point_3d_cam, scale, points_ori, bbox = pnp_shell(opt, meta, bbox,invert_points, size/size[1], OPENCV_RETURN=False)
    except:
        print("GT wrong point order")
        return [0]
    
    # return the 3D world points after PnP:
    return np.array(bbox["kps_3d_cam"])

   
def get_annotations():
    with open(root_json_gt, "r") as f:
        data = json.load(f)

    return data
   
def evaluate_img(root_json_detect, img_id, verbose=False):
    img_list = os.listdir(root_img)
    img_name = img_id + ".JPG"

    if img_name not in img_list:
        print("Image not found or is not a box")
        return 1,0
        
    annotations = get_annotations()
    annotation = None

    # Finding the groundtruth annotation:
    if 0 < img_id < len(annotations)-1: 
        annotation = annotations[img_id]
    else:
        print("no annotation for given file")

    img = plt.imread(root_img + img_name)

    if verbose: # uitgebreid
        img = plt.imread(root_img + img_name)
        plt.imshow(img)
        plt.show()

    # Finding the annotation found after inference:
    # check if the image has a debug file:
    detections = os.listdir(root_json_detect)
    if img_id + ".json" not in detections:
        print("no detection json found")
        return 1,0

    # open detection json:
    with open(root_json_detect + img_id + ".json", "r") as f:
        detection = json.load(f)
    
    if len(detection["objects"]) == 0:
        print("no detection")
        return 1, 0
    
    detection_points = np.array(detection["objects"][0]["kps_3d_cam"])
    
    # OPT:
    opt = opts()
    opt.nms = True
    opt.obj_scale = True
    opt.c = "cereal_box" # category
    
    # Meta: 
    camera_ford = np.array([[3648, 0, 2736], [0, 3648, 1824], [0, 0, 1]], dtype=np.float32)
    meta={"width": img.shape[1],"height": img.shape[0], "camera_matrix":camera_ford }
    
    # Get 3D GT:
    gt_points = get_gt_points(annotation, meta, opt)
    if len(gt_points) == 1:
        print("wrong annotation point order")
        return 1,0
 
    # Make box objects and determine IoU:
    gt_box = Boxcls(gt_points)
    detect_box = Boxcls(detection_points)

    iou = IoU(detect_box,gt_box) # calculate IoU:

    print("iou old= ", iou.iou())
    print(detect_box.vertices[0], gt_box.vertices[0])
    
    # Shift the box so it falls on middelpoint of detection:
    trans = gt_box.vertices[0] - detect_box.vertices[0]

    print(detect_box.vertices[0]+trans)
    
    translated = []
    for i in range(len(detect_box.vertices)):
        translated.append(detect_box.vertices[i]+trans)
    
    translated = np.array(translated)
    detect_box = Boxcls(translated)

    print("volume detected", detect_box.volume)
    print("volume GT",gt_box.volume)
    
    print("scale detected", detect_box.scale)
    print("scale GT",gt_box.scale)
    
    iou = IoU(detect_box,gt_box)
    result = iou.iou()
    print("IoU after shifting bbox", result)
    img = plt.imread(root_json_detect+img_id+"_out_kps_processed_pred.png")
    return result, img


def main():
    # Dir with detections:
    root_json_detect = "exp/"
    img_id = "3"
    args = sys.argv[1:]

    # Takes one argument: image id to evaluate
    if len(args)==0 or len(args)> 1:
        print("call function with one argument, e.g. '0587'")
        print("using default ")
    else:
        img_id = args[0]
    
    iou, img = evaluate_img(root_json_detect, img_id)
    if iou == 1:
        print("error see prints")
    
    else:
        iou *= 100
        print(f"the intersection of union was {round(iou)}, see the plot for the result")
        plt.imshow(img)
        plt.show(block = False)
        input("hit[enter] to end.")

    plt.close('all')
    

def find_iou(root_json_detect, file):
    with open(root_json_detect+file, "r") as f:
        data=json.load(f)

    if len(data["objects"])==0:
        print("no object found")
        return 0
        
    
    return data["objects"][0]["IOU"]

     
def get_statistics(dection_results, verbose=False):
    with open(root_json_gt+"order_train_size_correct.json", "r") as f:
        train=json.load(f)
    print(len(train))

    with open(root_json_gt+"order_val_size_correct.json", "r") as f:
        val=json.load(f)
    print(len(val)) 
    json_files=os.listdir(dection_results)
    print(len(json_files))
    total_train=0
    total_val=0
    missed_train=0
    missed_val=0
    correct_train=0
    correct_val=0
    correct_train_50=0
    correct_val_50=0
    gt_fail_train=0
    gt_fail_val=0
    fail=[]
    
    for file in json_files:
        name=file.split(".")[0]
        found=False
        for dict in train:
            if name in dict["image_name"]:
              
                found="train"
                break
        
            
        
        if found==False:
            for dict in val:
                if name in dict["image_name"]:
                   
                    found="val"
                    break
        # find iou and add statistics      
        iou=find_iou(dection_results, file)
        print(iou)
        if found=="val":
            total_val+=1
            if iou==None:
                gt_fail_val+=1
            elif iou==0:
                missed_val+=1
            else:
                correct_val+=iou
                if iou>=0.5:
                    correct_val_50+=1
                else:
                    fail.append(file)
                    
                    
        elif found=="train":
            total_train+=1
            if iou==None:
                gt_fail_train+=1
            elif iou==0:
                missed_train+=1
            else:
                correct_train+=iou
                if iou>=0.5:
                    correct_train_50+=1
                    
                else:
                    fail.append(file)
        else:
            print("json not in a file")
        
    print()
    print("TRAIN: ")
    print(f"The total number of samples is {total_train}")
    print(f"The total number of miss detections =  {missed_train}")
    print(f"The total number of miss GT =  {gt_fail_train}")
    print(f"The train average iou is {correct_train/(total_train-missed_train-gt_fail_train)}")
    print(f"The train 50% iou {correct_train_50/(total_train-missed_train-gt_fail_train)}")
    print()
    print("VALIDATION: ")
    print(f"The total number of samples is {total_val}")
    print(f"The total number of miss detections =  {missed_val}")
    print(f"The total number of miss GT =  {gt_fail_val}")
    print(f"The validation average iou is {correct_val/(total_val-missed_val-gt_fail_val)}")
    print(f"The val 50% iou {correct_val_50/(total_val-missed_val-gt_fail_val)}")
    
    print("failed: ", fail)
    
    if verbose:
        for name in fail:
            name=name.split(".")[0]+".JPG"
            img=plt.imread("/apollo/mle/Datasets/boxes/"+ name)
            plt.imshow(img)
            plt.show()


if __name__=="__main__":
    # get_statistics("demo/green_background/")
    main()