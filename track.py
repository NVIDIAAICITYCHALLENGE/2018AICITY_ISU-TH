import numpy as np
import os
import sys
import cv2
import pickle
import time

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    if xB<=xA or yB<=yA:
        iou = 0
    else:
        interArea = (xB - xA + 1) * (yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# reduce duplicated bbox using iou threshold
def remove_duplicated_candidate(candidate, iou_threshold):
    iou_res=[]
    for i in range(len(candidate)):
        for j in range(i+1,len(candidate)):
            iou_res.append([i, j, bb_intersection_over_union(candidate[i], candidate[j])])

    overlap=[(bbs[0], bbs[1]) for bbs in iou_res if bbs[2]>iou_threshold]   
    unique_overlap=list(x for l in overlap for x in l)
    keep=[bbs[0] for bbs in iou_res if bbs[2]>iou_threshold]
    drop=[bbs[1] for bbs in iou_res if bbs[2]>iou_threshold]
    new_keep = [k for k in keep if k not in drop]
    new_candidate = [candidate[n] for n in range(len(candidate)) if n not in unique_overlap or n in new_keep]

    return new_candidate

#remove largest and smallest
def reduce_candidate(new_candidate):
    outliers=[]
    for i in range(len(new_candidate)):
        box = new_candidate[i]
        area = (box[2]-box[0])*(box[3]-box[1])
        if area>=70000 or area<400:
            outliers.append(i)

    reduced_candidate = [new_candidate[r] for r in range(len(new_candidate)) if r not in outliers]
    
    return reduced_candidate

# process the bbox from raw detection to clear detection
def detection_process(test_box, test_score, test_class, score_threshold, iou_threshold):
    test_box = np.squeeze(test_box)
    test_score = np.squeeze(test_score)
    test_class = np.squeeze(test_class) # 3,6,8
    
    # use score threshold to pick up detections
    candidate = [(int(box[1]*1920), int(box[0]*1080), int(box[3]*1920), int(box[2]*1080)) 
          for box, score, cls in zip(test_box, test_score, test_class) 
                 if score > score_threshold and (cls==3 or cls==6 or cls==8)]
    
    # use iou threshold to filter out overlap detections
    new_candidate = remove_duplicated_candidate(candidate, iou_threshold)
    reduced_candidate = reduce_candidate(new_candidate)
    
    return reduced_candidate

#roi, expand box itself
def get_roi(target_box):
    h = target_box[3]-target_box[1]
    w = target_box[2]-target_box[0]

    if target_box[0]-w<0:
        p1x=0
    else:
        p1x=target_box[0]-w
    if target_box[2]+w>1920:
        p2x=1920
    else:
        p2x=target_box[2]+w

    if target_box[1]-h<0:
        p1y=0
    else:
        p1y=target_box[1]-h
    if target_box[3]+h>1080:
        p2y=1080
    else:
        p2y=target_box[3]+h
    
    return (p1x, p1y, p2x, p2y) # roi


def in_roi(roi, box):
    xA = max(roi[0], box[0])
    yA = max(roi[1], box[1])
    xB = min(roi[2], box[2])
    yB = min(roi[3], box[3])
    
    if xB<=xA or yB<=yA:
        interArea = 0
    else:
        interArea = (xB - xA + 1) * (yB - yA + 1)
    in_rate = interArea/((box[2]-box[0])*(box[3]-box[1]))
    
    return in_rate


def find_cars(frame2, bbox2, roi):
    cars_index=[]
    for i in range(len(bbox2)):
        if in_roi(roi, bbox2[i][3])>0.8:
            cars_index.append(i) 
    cars_box = []
    cars_img = []
    if len(cars_index)!=0:
        for i in cars_index:
            cars_box.append(bbox2[i][3])    
            cars_img.append(frame2[bbox2[i][3][1]:bbox2[i][3][3], bbox2[i][3][0]:bbox2[i][3][2]])
            
    return cars_box, cars_img

def compare_find(target_img, cars_img):
    
    tar = cv2.calcHist([target_img], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    tar = cv2.normalize(tar, tar).flatten()
    s_min = 10000
    
    for ind in range(len(cars_img)):
        hist = cv2.calcHist([cars_img[ind]], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        similiarity = cv2.compareHist(tar, hist, cv2.HISTCMP_CHISQR) #smaller, better
        #CORREL #lager, better
        #INTERSECT #lager, better
        #BHATTACHARYYA #smaller, better
        if similiarity<1:
            if similiarity<s_min:
                s_min = similiarity
                found_ind = ind

    if s_min==10000:
        found_ind = 9999
            
    return found_ind

def initial_numbering(bbox_start, frame_id):
    numbered_start=[]
    ids=range(len(bbox_start))
    for id in ids:
        numbered_start.append((video_id, frame_id, id, bbox_start[id]))
    global_id=id
    return numbered_start, global_id

def initial_numbering_bbox2(bbox2, frame_id):
    new_bbox2=[]
    for box in bbox2:
        new_bbox2.append((video_id, frame_id, -1, box))
    return new_bbox2

def track(frame1, frame2, bbox1, bbox2, global_id):   
    
    for item in bbox1:

        target_box = item[3] # 1~index of box, 3~box array
        object_id = item[2] # 1~index of box, 2~object_id

        target_img = frame1[target_box[1]:target_box[3], target_box[0]:target_box[2]]

        roi = get_roi(target_box)

        cars_box, cars_img = find_cars(frame2, bbox2, roi)

        found_ind = compare_find(target_img, cars_img)

        if found_ind==9999:        
            found_car_box = []
        else:
            found_car_box = cars_box[found_ind]
            
            #find match and replace object id in bbox2
            for i in range(len(bbox2)):
                if sum(found_car_box==bbox2[i][3])==4:
                    bbox2[i]=(bbox2[i][0], bbox2[i][1], object_id, bbox2[i][3])
            
    # no object id, assign global id
    for i in range(len(bbox2)):        
        if bbox2[i][2]==-1:
            bbox2[i]=(bbox2[i][0], bbox2[i][1], global_id+1, bbox2[i][3])
            global_id += 1

    
    return bbox1, bbox2, global_id

if __name__ == '__main__':

	vid_match_table =  {'Loc1_1':1,
						'Loc1_2':2,
						'Loc1_3':3,
						'Loc1_4':4,
						'Loc1_5':5,
						'Loc1_6':6,
						'Loc1_7':7,
						'Loc1_8':8,
						'Loc2_1':9,
						'Loc2_2':10,
						'Loc2_3':11,
						'Loc2_4':12,
						'Loc2_5':13,
						'Loc2_6':14,
						'Loc2_7':15,
						'Loc2_8':16,
						'Loc3_1':17,
						'Loc3_2':18,
						'Loc3_3':19,
						'Loc3_4':20,
						'Loc3_5':21,
						'Loc3_6':22,
						'Loc4_1':23,
						'Loc4_2':24,
						'Loc4_3':25,
						'Loc4_4':26,
						'Loc4_5':27}


	detect_p_dir = 'all_p/detect_p/'
	track_p_dir = 'all_p/track_p/'
	detect_p_list = np.sort(os.listdir(detect_p_dir))
	for detect_p in detect_p_list:
		vid_name = detect_p.split('.')[0]
		video_id = vid_match_table[vid_name]

		(all_boxes,all_scores,all_classes) = pickle.load( open( detect_p_dir+detect_p, "rb" ) )

		path = 'track1_frames/' + vid_name

		bbox=[]
		for test in zip(all_boxes, all_scores, all_classes):
		    bbox.append(detection_process(test[0], test[1], test[2], score_threshold=0.2, iou_threshold=0.3))

		new_bbox=[]
		count = 0 
		for x in range(1800-1):
		    count += 1
		    since = time.time()
		    # print(path+'/image%s.jpg'%str(x+1))
		    frame1=cv2.cvtColor(cv2.imread(path+'/image%s.jpg'%str(x+1)), cv2.COLOR_BGR2RGB)
		    frame2=cv2.cvtColor(cv2.imread(path+'/image%s.jpg'%str(x+2)), cv2.COLOR_BGR2RGB)
		    bbox1 = np.copy(bbox[x])
		    bbox2 = np.copy(bbox[x+1])

		    if x == 0:
		        bbox1, global_id = initial_numbering(bbox1, frame_id=x+1)
		        bbox2 = initial_numbering_bbox2(bbox2, frame_id=x+2)
		        new_bbox.append(bbox1)
		        
		        
		    else:
		        bbox2 = initial_numbering_bbox2(bbox2, frame_id=x+2)
		        bbox1 = new_bbox[x]

		    _, bbox2, global_id = track(frame1, frame2, bbox1, bbox2, global_id)
		 
		    new_bbox.append(bbox2)

		    print('{}/1799 {}  {}'.format(count,time.time()-since,vid_name))

		max_obj=0
		for item in new_bbox[-1]:
		    if item[2]>max_obj:
		        max_obj=item[2]


		all_object=[]
		for o in range(max_obj):
		    objecto=[]
		    for i in range(len(new_bbox)):
		        for box in new_bbox[i]:
		            # get object number o
		            if box[2]==o:
		                objecto.append(box)
		    all_object.append(objecto)


		pickle.dump( all_object, open( track_p_dir+vid_name+"_all_object.p", "wb" ) )