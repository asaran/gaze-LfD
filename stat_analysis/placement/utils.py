import cv2
import ast
from bisect import bisect_left
from numpy import ones,vstack
from numpy.linalg import lstsq
import numpy as np
import math
import rosbag
import math
import gzip
import os


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """   
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], 0
    if pos == len(myList):
        return myList[-1], len(myList)-1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after, pos
    else:
       return before, pos-1


# returns a list of rgb color values for gaze point for each video frame
def get_color_timeline(data, video_file):
    timeline = []
    fixations = {'red': [], 'yellow':[], 'green':[], 'other':[]}
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                vid2ts[d['vts']] = d['ts']

        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')

    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))
        a = all_vts[i]


    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds

    while success:          
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts,  _ = takeClosest(all_ts,ts)
        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        #h,s,v = img[gaze_coords[1]][gaze_coords[0]]
        b, g, r = img[gaze_coords[1]][gaze_coords[0]]
        instant_color = [r/255.0,g/255.0,b/255.0]
        timeline.append(instant_color)

        last_gaze_pt = gaze_coords

        count += 1
        success, img = vidcap.read()



    vidcap.release()
    cv2.destroyAllWindows()
    return timeline


def get_cumulative_gaze_dist(data, video_file):
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                vid2ts[d['vts']] = d['ts']

        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')

    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))
        a = all_vts[i]


    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    print('reading video file')

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    gaze_pts = []

    current_dist = 0
    cumulative_dist = [0]
    tracker_ts, _ = takeClosest(all_ts,all_vts[0])
    gx_p, gy_p = gp[tracker_ts]

    while success:  
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts, _ = takeClosest(all_ts,ts)

        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        gaze_pts.append(gaze_coords)

        gx, gy = gaze_coords
        d = math.sqrt(math.pow(gx-gx_p,2)+math.pow(gy-gy_p,2))
        current_dist = current_dist + d
        cumulative_dist.append(current_dist)
        gx_p, gy_p = gx, gy
        
        count += 1
        success, img = vidcap.read()

    vidcap.release()
    cv2.destroyAllWindows()

    return cumulative_dist

def get_color_name(hsv):

    color_ranges = {
        'red':   [[160,90,30],[190,190,135]],
        'green': [[83,55,25],[115,110,125]],
        'yellow': [[5,100,40],[35,165,170]],
        'blue': [[95,75,55],[120,120,160]],
        'orange': [[0,145,65],[190,230,175]],
        'purple': [[115,50,10],[145,150,125]],
        'black': [[0,0,0],[180,255,40]],
        'white': [[0,0,170],[180,255,255]]
    }

    color_val = {
        'black': (0,0,0),
        'white': (255,255,255),
        'red': (0,0,255),
        'green': (0,255,0),
        'yellow': (0,255,255),
        'blue': (255,0,0),
        'orange': (165,0,255),
        'purple': (32,240,160)
    }
        

    h,s,v = hsv
    color = ''
    value = None
    for i, (n,r) in enumerate(color_ranges.items()):
        if h>=r[0][0] and h<=r[1][0]:
            if s>=r[0][1] and s<=r[1][1]:
                if v>=r[0][2] and v<=r[1][2]:
                    color = n 
                    value = color_val[n]

    pasta_color_range = [[0,30,0],[40,130,150]]
    p = pasta_color_range
    if color=='':
        if h>=p[0][0] and h<=p[1][0]:
            if s>=p[0][1] and s<=p[1][1]:
                if v>=p[0][2] and v<=p[1][2]:
                    color = 'pasta'
                    value = color_val['yellow']

    return color, value


def get_color_name_from_hist(gaze_coords, img_hsv, radius):
    color_hist ={
        'orange': 0,
        'yellow': 0,
        'red': 0,
        'green': 0,
        'black': 0,
        'pasta': 0,
        'blue': 0,
        'purple': 0,
        'other': 0
    }

    color_val = {
        'black': (0,0,0),
        'red': (0,0,255),
        'green': (0,255,0),
        'yellow': (0,255,255),
        'pasta': (0,255,255),
        'blue': (255,0,0),
        'orange': (165,0,255),
        'purple': (32,240,160),
        'other': (192,192,192)
    }

    x, y = gaze_coords
    hsv = img_hsv[y][x]
    h,s,v = hsv
    color = ''
    value = None

    # pixels in the image which lie inside a circle of given radius
    min_x, max_x = max(0,x-radius), min(1920, x+radius)
    min_y, max_y = max(0,y-radius), min(1080, y+radius)
    for i,j in zip(range(min_x,max_x), range(min_y,max_y)):
        d = math.pow((i-x),2)+ math.pow((j-y),2)
        if d<= math.pow(radius,2):
            curr_hsv= img_hsv[j][i]
            current_color, _ = get_color_name(curr_hsv)
            if current_color in color_hist.keys():
                color_hist[current_color] += 1
            else:
                color_hist['other'] += 1

    max_val = 0
    max_color = ''
    for key,val in color_hist.items():
        if val>max_val:
            max_val = val
            max_color = key


    # do not assign other color if relevant colors are present
    second_max_val = 0
    second_max_color = ''
    if max_color=='other':
        for key,val in color_hist.items():
            if key=='other':
                continue
            else:
                if val>second_max_val:
                    second_max_val = val
                    second_max_color = key
        if second_max_val>5:
            max_color = second_max_color
            max_val = second_max_val

    value = color_val[max_color]
    return max_color, value



def find_saccades(gaze_pts, fps):
    speed = []
    saccade_indices = []
    speed.append(0)
    dt = 1.0/fps
    for i in range(1,len(gaze_pts)):
        g = gaze_pts[i]
        prev_g = gaze_pts[i-1]
        s = (math.sqrt(math.pow(g[0]-prev_g[0],2)+math.pow(g[1]-prev_g[1],2)))/dt
        if s>800:
            saccade_indices.append(i)
    return saccade_indices



# returns a list of rgb color values for gaze point for each video frame
def get_hsv_color_timeline(data, video_file):
    timeline = []
    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                #vid_time = d['ts'] - d['vts']
                vid2ts[d['vts']] = d['ts']

        # TODO: if multiple detections for same time stamp?
        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')


    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))
        a = all_vts[i]

    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    print('reading video file')

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    gaze_pts = []

    while success:    
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)   
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts,  _ = takeClosest(all_ts,ts)
        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))
        gaze_pts.append(gaze_coords)

        h,s,v = img_hsv[gaze_coords[1]][gaze_coords[0]]
        instant_color = [h, s, v]
        timeline.append(instant_color)

        count += 1
        success, img = vidcap.read()

        

    vidcap.release()
    cv2.destroyAllWindows()

    saccade_indices = []
    saccade_indices = find_saccades(gaze_pts, fps)

    return timeline, saccade_indices


# returns a list of frame indices corresponding to the annotated KF for video demonstrations
def get_video_keyframes(user_id, seg, video_file, video_kf_file):
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('read video file')

    vidcap.release()
    cv2.destroyAllWindows()

    # read video files
    with open(video_kf_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    # print(content)
    print('read text file')

    # find segmentation points in video file
    keyframes = {
        'Start': [],
        'Open': [],
        'Stop': []
    }

    kf_type = {
        1: 'Start',
        2: 'Open',
        3: 'Stop',
        4: 'Start',
        5: 'Open',
        6: 'Stop'
    }

    if seg%2==0:
        r = [4, 5, 6]
    else:
        r = [1, 2, 3] 
    for kf in content:                  
        data = kf.split(' ')
        # print(data)
        user = data[0]
        if(user == user_id):
            for i in r:
                d = data[i]
                # print(d)
                if(d=='end'):
                    frame_idx = length
                else:
                    kf_time = float(d)
                    frame_idx = math.floor(kf_time*fps)
                k = kf_type[i]
                keyframes[k].append(int(frame_idx))

    print('Found start and stop keyframe indices')
    return keyframes

def read_json(data_dir):
    data = []
    files = os.listdir(data_dir)
    
    for file in files:
        if (file.endswith("json.gz")):
            with gzip.open(data_dir+'/'+file, "rb") as f:
                data=f.readlines()
            
                for r in range(len(data)):
                    row = data[r]
                    data[r] = ast.literal_eval(row.strip('\n'))

    vid2ts = {}     # dictionary mapping video time to time stamps in json
    right_eye_pd, left_eye_pd, gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

    for d in data:
        if 'vts' in d and d['s']==0:
            if d['vts'] == 0:
                vid2ts[d['vts']] = d['ts']
            else:
                vid2ts[d['vts']] = d['ts']

        if 'pd' in d and d['s']==0 and d['eye']=='right':
            right_eye_pd[d['ts']] = d['pd']
        if 'pd' in d and d['s']==0 and d['eye']=='left':
            left_eye_pd[d['ts']] = d['pd']

        if 'gp' in d and d['s']==0 :
            gp[d['ts']] = d['gp']   #list of 2 coordinates
    print('read json')

    # map vts to ts
    all_vts = sorted(vid2ts.keys())
    a = all_vts[0]
    model = []
    for i in range(1,len(all_vts)):
        points = [(a,vid2ts[a]),(all_vts[i],vid2ts[all_vts[i]])]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        model.append((m,c))

    return data, gp, model, all_vts


def filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx):
    print('filtering fixations')
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    KT_fixation_count = {
        'red': 0,
        'orange': 0,
        'blue': 0,
        'green': 0,
        'black': 0,
        'yellow': 0,
        'purple': 0,
        'other': 0,
        'pasta': 0
    }

    fixation_count = KT_fixation_count

    all_ts = sorted(gp.keys())
    total_count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    window = []
    win_size = 3
    radius = 100
    valid_count = 0
    while success:  
        if total_count<start_idx or total_count>end_idx:
            total_count += 1
            success, img = vidcap.read()
            continue

        frame_ts = int((total_count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c

        tracker_ts,_ = takeClosest(all_ts,ts)

        gaze = gp[tracker_ts]
        gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        color_name, color_value = get_color_name_from_hist(gaze_coords, img_hsv, radius)
        window.append(color_name)
        if(len(window)>win_size):
            del window[0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        if total_count not in saccade_indices:
            # might be a fixation
            fixation = True
            for det_c in window:
                if det_c!=color_name:
                    fixation=False
            if(fixation):
                fixation_count[color_name] += 1

        valid_count += 1
        total_count += 1
        success, img = vidcap.read()

    cv2.destroyAllWindows()
    print total_count
    for f in fixation_count:
        fixation_count[f] = fixation_count[f]*100.0/valid_count

    return fixation_count



def get_kt_keyframes(all_vts, model, gp, video_file, bag_file):
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, img = vidcap.read()
    print('reading video file')

    last_fixation_color =(0,0,0)
    all_ts = sorted(gp.keys())
    count = 0
    imgs = []       # list of image frames
    frame2ts = []   # corresponding list of video time stamp values in microseconds
    videoframe2trackerts = []
    gaze_pts = []

    while success:          
        frame_ts = int((count/fps)*1000000)
        frame2ts.append(frame_ts)

        less = [a for a in all_vts if a<=frame_ts]
        idx = len(less)-1

        if idx<len(model):
            m,c = model[idx]
        else:
            m,c = model[len(model)-1]
        ts = m*frame_ts + c
        tracker_ts, _ = takeClosest(all_ts,ts)
        videoframe2trackerts.append(tracker_ts)

        count += 1
        success, img = vidcap.read()

    vidcap.release()
    cv2.destroyAllWindows()

    # find segmentation points on bagfile
    all_keyframe_indices = []
    record_k = False
    bag = rosbag.Bag(bag_file)
    print(bag_file)
    if bag.get_message_count('/gaze_tracker')!=0:       # gaze_tracker topic was recorded
        for topic, msg, t in bag.read_messages(topics=['/gaze_tracker','/log_KTframe']):
            if (topic=='/log_KTframe'):
                if("Recorded keyframe" in msg.data):
                    record_k = True

                    kf_type = 'Other'

                if("Open" in msg.data):
                    record_k = True
                    kf_type = 'Open'

            if (topic == '/gaze_tracker'):
                if(record_k == True):                   
                    if('gp' in msg.data):                   
                        gaze_msg = msg.data
                        s = gaze_msg.find('"ts":')
                        e = gaze_msg.find(',')
                        gaze_ts = gaze_msg[s+5:e]
                        tracker_ts, frame_idx = takeClosest(videoframe2trackerts,int(gaze_ts))
                        all_keyframe_indices.append(frame_idx)
                        record_k = False
    bag.close()
    return all_keyframe_indices