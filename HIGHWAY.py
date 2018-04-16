
import numpy as np
import cv2
import pickle
from sklearn import preprocessing

def convert(obj, location_id, location_number):
    pts = np.array([[[obj[3][2], obj[3][3]]]],dtype="float32")
    m = cv2.perspectiveTransform(pts, warpM[location_id][location_number])
    return (m[0][0][0], m[0][0][1])

def distance(a, b):
    distance = np.sqrt((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1]))
    return distance

def to_mph(pixel_distance, location_id, location_number):
    
    ### be aware! In warping process, Location 1 is using 3 lanes and Location 2 is just using 1 lane!
    
    if location_id == 1:
        num_lanes = 3
    else:
        num_lanes = 1

    pps=np.float(pixel_distance)*30

    mph=np.float(pps)/conversion[location_id][location_number][0]*12*num_lanes/5280*3600
    
    return mph

def ma_smooth(data, window_width=5):   
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    smoothed = np.append(ma_vec, np.repeat(ma_vec[-1],4))
    return smoothed

def highway_speed(all_object, location_id, location_number):
    # raw speed
    all_speed=[]
    for objecto in all_object:
        trace=[]
        for i in range(len(objecto)):
            m = convert(objecto[i], location_id, location_number)
            trace.append((objecto[i], m))
        speedo=[]
        for i in range(len(trace)-1):
            pixel_distance = distance(trace[i][1],trace[i+1][1])
            speed = to_mph(pixel_distance, location_id, location_number)
            speedo.append((trace[i][0], speed))
            if i==len(trace)-2:
                speedo.append((trace[i+1][0], speed))
        all_speed.append(speedo)
    # get mean
    all_mean=[]
    for i in range(len(all_speed)):
        if len(all_speed[i])>0:
            temp=[]
            all_remain=[]
            for speed in all_speed[i]:
                ind_speed = speed[1]
                remain = speed[0]
                temp.append(ind_speed)
                all_remain.append(remain)
            obj_mean = np.mean(temp)
            all_mean.append(obj_mean)
    all_new_mean = conversion[location_id][location_number][1]+preprocessing.scale(all_mean)
    # get final
    all_noise=[]
    all_new_remain=[]
    for i in range(len(all_speed)):
        if len(all_speed[i])>0:
            temp=[]
            all_remain=[]
            for speed in all_speed[i]:
                ind_speed = speed[1]
                remain = speed[0]
                temp.append(ind_speed)
                all_remain.append(remain)
            noise = preprocessing.scale(temp)
            if len(noise)>=5:
                noise = ma_smooth(noise)
            all_noise.append(noise)
            all_new_remain.append(all_remain)
    final=[]
    for i in range(len(all_new_remain)):
        for j in range(len(all_new_remain[i])):
            final.append((all_new_remain[i][j], all_new_mean[i]+all_noise[i][j]))
            
    return final

conversion = pickle.load(open("all_p/conversion1.p", "rb"))

warpM = pickle.load(open("all_p/warpM1.p", "rb"))

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

for i in range(1,3):
    for j in range(1,9):
        location_id = i
        location_number = j
        vid_name = 'Loc%s_%s' %(location_id,location_number)
        vid = vid_match_table[vid_name]

        test = pickle.load(open("all_p/track_p/"+vid_name+"_all_object.p", "rb" ))

        new_test=[]
        for obj in test:
            new_obj=[]
            for item in obj:
                if item[3][1]>=conversion[location_id][location_number][2]:
                    new_obj.append(item)
            if len(new_obj)>0:
                new_test.append(new_obj)

        final = highway_speed(new_test, location_id, location_number)


        txtfile = open('res/video%s_final.txt' %vid, 'w')

        for obj in final:
            txtfile.write("%d %d %d " % (obj[0][0],obj[0][1],obj[0][2]))
            for a in obj[0][3]:
                txtfile.write("%d " % a)
            txtfile.write("%.2f %d\n" %(obj[1],0))
                
        txtfile.close()