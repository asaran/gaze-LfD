from numpy import ones,vstack
from numpy.linalg import lstsq
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import ast 
from bisect import bisect_left
import os
import gzip
import math
import rosbag

class LfdData():
    def __init__(self, exp, demo_type, short_video=False):
        self.data_dir = '../data/ut-lfd/'+ exp + '/' #'../data/pouring/experts/KT6/5fyyvco/segments/6/'
        self.visualize = False
        self.users = []
        self.user_dir = ''
        self.gp = {}
        self.vid2ts = {}
        self.all_vts = []
        self.model = []
        self.vid_start = {}
        self.vid_end = {}
        self.short_video = short_video
        # self.reach, self.grasp, self.trans, self.pour, self.ret, self.vid_rel = {}, {}, {}, {}, {}, {}
        self.seg_times, self.segs = {}, {}
        self.labels = { 'Reaching': 0,
                        'Grasping': 1,
                        'Transport': 2,
                        'Pouring': 3,
                        'Return': 4,
                        'Release': 5}
        self.order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

        self.img_count = 0

        if demo_type == 'v':
            f = open(self.data_dir +"images.txt",'w')
            f2 = open(self.data_dir +"image_class_labels.txt","w")
            f3 = open(self.data_dir +"image_gaze.txt","w")
            
            f.close()
            f2.close()
            f3.close()

            ft, fv = [], []
            for i in range(10):
                ft = open(self.data_dir +"train"+str(i)+".list","w")
                fv = open(self.data_dir +"val"+str(i)+".list","w")
                ft.close()
                fv.close()

        if demo_type=='k' and not self.short_video:
            f = open(self.data_dir +"KT/images.txt",'w')
            f2 = open(self.data_dir +"KT/image_class_labels.txt","w")
            f3 = open(self.data_dir +"KT/image_gaze.txt","w")

            f.close()
            f2.close()
            f3.close()

            ft, fv = [], []
            for i in range(10):
                ft = open(self.data_dir +"KT/train"+str(i)+".list","w")
                fv = open(self.data_dir +"KT/val"+str(i)+".list","w")
                ft.close()
                fv.close()

        if demo_type=='k' and self.short_video:
            f = open(self.data_dir +"KT_50/images.txt",'w')
            f2 = open(self.data_dir +"KT_50/image_class_labels.txt","w")
            f3 = open(self.data_dir +"KT_50/image_gaze.txt","w")

            f.close()
            f2.close()
            f3.close()

            ft, fv = [], []
            for i in range(10):
                ft = open(self.data_dir +"KT_50/train"+str(i)+".list","w")
                fv = open(self.data_dir +"KT_50/val"+str(i)+".list","w")
                ft.close()
                fv.close()


    def read_json(self, my_dir):
        data_file = my_dir+"livedata.json.gz"
        with gzip.open(data_file, "rb") as f:
            data=f.readlines()

        for r in range(len(data)):
            row = data[r]
            data[r] = ast.literal_eval(row.strip('\n'))

        self.vid2ts = {}     # dictionary mapping video time to time stamps in json
        right_eye_pd, left_eye_pd, self.gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

        for d in data:
            if 'vts' in d and d['s']==0:
                if d['vts'] == 0:
                    self.vid2ts[d['vts']] = d['ts']
                else:
                    self.vid2ts[d['vts']] = d['ts']

            # TODO: if multiple detections for same time stamp?
            if 'pd' in d and d['s']==0 and d['eye']=='right':
                right_eye_pd[d['ts']] = d['pd']
            if 'pd' in d and d['s']==0 and d['eye']=='left':
                left_eye_pd[d['ts']] = d['pd']

            if 'gp' in d and d['s']==0 :
                self.gp[d['ts']] = d['gp']   #list of 2 coordinates
        print('read json')

        # map vts to ts
        self.all_vts = sorted(self.vid2ts.keys())
        a = self.all_vts[0]
        self.model = []
        for i in range(1,len(self.all_vts)):
            points = [(a,self.vid2ts[a]),(self.all_vts[i],self.vid2ts[self.all_vts[i]])]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]
            # print("Line Solution is ts = {m}vts + {c}".format(m=m,c=c))
            self.model.append((m,c))


    
    def takeClosest(self, myList, myNumber):
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


    def gaze_for_video_imgs(self, seg, user, trial, skip):
        user_dir = self.data_dir + 'videos/' + user + '/' + seg + '/segments/' + str(trial) + '/'
        self.read_json(user_dir)
        vidcap = cv2.VideoCapture(user_dir+'fullstream.mp4')
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_time = frame_count/fps
        # print fps     #25 fps
        success, img = vidcap.read()

        if self.visualize:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(user_dir+'gaze_overlayed.avi',fourcc,fps,(1920,1080))

        all_ts = sorted(self.gp.keys())
        count = 0
        imgs = []       # list of image frames
        frame2ts = []   # corresponding list of video time stamp values in microseconds

        # print(type(fps),type(self.vid_start[user]))
        start = abs(fps*self.vid_start[user])
        if self.vid_end[user]=='end':
            end = abs(fps*end_time)
        else:
            end = abs(fps*self.vid_end[user])

        # open files for image names and labels
        f = open(self.data_dir +"images.txt",'a')
        f2 = open(self.data_dir +"image_class_labels.txt","a")
        f3 = open(self.data_dir +"image_gaze.txt","a")
        ft, fv = [], []
        for i in range(10):
            f4 = open(self.data_dir +"train"+str(i)+".list","a")
            f5 = open(self.data_dir +"val"+str(i)+".list","a")

            ft.append(f4)
            fv.append(f5)

        while success:  
            # print(count)
            if count>=start and count<=end:
                frame_ts = int((count/fps)*1000000)
                frame2ts.append(frame_ts)

                less = [a for a in self.all_vts if a<=frame_ts]
                idx = len(less)-1

                if idx<len(self.model):
                    m,c = self.model[idx]
                else:
                    m,c = self.model[len(self.model)-1]
                ts = m*frame_ts + c

                tracker_ts, _ = self.takeClosest(all_ts,ts)

                gaze = self.gp[tracker_ts]
                gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

                if self.visualize:
                    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                    hsv = img_hsv[gaze_coords[1]][gaze_coords[0]]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color_name, color_value = get_color_name(hsv)
                    
                    if(color_name!=''):
                    #   print(color_name)
                        cv2.putText(img, color_name, (1430, 250), font, 1.8, color_value, 5, cv2.LINE_AA)

                    # print(hsv)
                    cv2.putText(img, str(hsv), (230, 250), font, 1.8, (255, 255, 0), 5, cv2.LINE_AA)

                    cv2.circle(img,gaze_coords, 25, (255,255,0), 3)
                    video.write(img)

                if(count%skip==0):
                    # write images
                    self.img_count+=1
                    img_name = user+'_'+str(trial)+'_'+str(count)+'.jpg'
                    cv2.imwrite(self.data_dir+'/VD_images/'+img_name, img)
                    f.write(str(self.img_count)+' '+img_name+'\n')
                    seg = self.find_segment(user, count, fps, end_time)
                    f2.write(str(self.img_count)+' '+str(self.labels[seg])+'\n')
                    # TODO: write normalized gaze coordinates
                    f3.write(str(self.img_count)+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

                    # if(user=='KT19' or user=='KT20'):
                    #   f5.write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')
                    # else:
                    #   f4.write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')
                    
                    fold_no = int(math.ceil(float(user[2:])/2))
                    print('user: '+user+'\tfold_no: '+str(fold_no))
                    fv[fold_no-1].write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

                    # this user's data goes to all other training folds
                    for i in range(10):
                        if i==fold_no-1:
                            continue
                        ft[i].write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

            count += 1
            success, img = vidcap.read()

        if self.visualize:
            cv2.destroyAllWindows()
            video.release()

        f.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
    
    def read_video_annotation_file(self):
        file_path = self.data_dir+'video_kf.txt'
        file = open(file_path, 'r') 

        for line in file:
            entries = line.strip('\n').split(' ')
            user = entries[0]
            self.users.append(user)
            self.vid_start[user] = float(entries[2])
            if entries[-1]=='end':
                self.vid_end[user] = entries[-1]  
            else:
                # print('last entry:', entries[-1])
                self.vid_end[user] = float(entries[-1])

            self.seg_times[user] = [float(e) if e!='end' else e for e in entries[2:-1] ]
            self.segs[user] = {}
            self.segs[user][float(entries[2])] = 'Reaching' 
            self.segs[user][float(entries[8])] = 'Reaching'
            self.segs[user][float(entries[3])] = 'Grasping'
            self.segs[user][float(entries[9])] = 'Grasping'
            self.segs[user][float(entries[4])] = 'Transport'
            self.segs[user][float(entries[10])] = 'Transport'
            self.segs[user][float(entries[5])] = 'Pouring'
            self.segs[user][float(entries[11])] = 'Pouring'
            self.segs[user][float(entries[6])] = 'Return'
            self.segs[user][float(entries[12])] = 'Return'
            self.segs[user][float(entries[7])] = 'Release' 
            if entries[13]!='end':
                self.segs[user][float(entries[13])] = 'Release' 
            else:
                self.segs[user][entries[13]] = 'Release' 

    def find_segment(self, user, count, fps, end_time):
        # print(self.segs[user])
        # print(user)
        for i in range(len(self.seg_times[user])-1):
            s = self.seg_times[user][i]
            s_next = self.seg_times[user][i+1]
            t = s
            if s_next=='end':
                t_next = end_time
            else:
                t_next = s_next
            # print('t: ',str(t))
            if count>=t*fps and count<=t_next*fps:
                return self.segs[user][s]
        s = self.seg_times[user][i+1]
        return self.segs[user][s]

    def create_video_imgs(self):
        self.read_video_annotation_file()
        # print(self.users)
        for user in self.users:
            print(user)
            exps = self.order[user]
            demo_type = 'v'
            if demo_type == exps[0]:
                trial = 1
            else:
                trial = 4

            user_dir = self.data_dir + 'videos/' + user + '/'
            d = os.listdir(user_dir)
            assert(len(d)==1)
            seg = d[0]
            # print(seg)
            # trial = 4
            self.gaze_for_video_imgs(seg, user, trial, 2)


    def create_kt_imgs(self):

        self.users = self.order.keys()
        # self.users = ['KT2', 'KT9', 'KT10', 'KT11', 'KT12', 'KT13','KT15']
        for user in self.users:
            print(user)
            exps = self.order[user]
            demo_type = 'k'
            if demo_type == exps[0]:
                trials = [1, 2, 3]
                # trials = [1]
            else:
                trials = [4, 5, 6]
                # trials = [5]

            user_dir = self.data_dir + 'videos/' + user + '/'
            d = os.listdir(user_dir)
            assert(len(d)==1)
            seg = d[0]
            # print(seg)
            # trial = 4
            for trial in trials:
                if not self.short_video:
                    self.gaze_for_kt_imgs(seg, user, trial, 5)
                else:
                    self.gaze_for_kt_imgs_short(seg, user, trial)


    def gaze_for_kt_imgs(self, seg, user, trial, skip):
        user_dir = self.data_dir + 'videos/' + user + '/' + seg + '/segments/' + str(trial) + '/'
        if not os.path.exists(user_dir):
            return
        bag_dir = '/home/akanksha/Documents/gaze_for_lfd_study_data/gaze_lfd_user_study/' 

        self.read_json(user_dir)
        vidcap = cv2.VideoCapture(user_dir+'fullstream.mp4')
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_time = frame_count/fps
        # print fps     #25 fps
        success, img = vidcap.read()

        if self.visualize:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(user_dir+'gaze_kt_overlayed.avi',fourcc,fps,(1920,1080))

        # open files for image names and labels
        f = open(self.data_dir +"KT/images.txt",'a')
        f2 = open(self.data_dir +"KT/image_class_labels.txt","a")
        f3 = open(self.data_dir +"KT/image_gaze.txt","a")
        ft, fv = [], []
        for i in range(10):
            f4 = open(self.data_dir +"KT/train"+str(i)+".list","a")
            f5 = open(self.data_dir +"KT/val"+str(i)+".list","a")

            ft.append(f4)
            fv.append(f5)


        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)

        bag_file = ''
        
        for file in bagfiles:
            # if (file.endswith("kt-p1.bag")):
            #     bag_file = bagloc + file
            if (file.endswith("kt-p1.bag") and (int(trial)==1 or int(trial)==4)):
                bag_file = bagloc + file
            elif (file.endswith("kt-p2.bag") and (int(trial)==2 or int(trial)==5)):
                bag_file = bagloc + file
            elif (file.endswith("kt-p3.bag") and (int(trial)==3 or int(trial)==6)):
                bag_file = bagloc + file
        
        if bag_file == '':
            print('Bag file does not exist for KT demo, skipping...')
            return
        
        # read bag file
        video_file = user_dir+'fullstream.mp4'
        keyframes, keyframe_indices = self.get_kt_keyframes_labels(video_file, bag_file)

        first_grasp = False
        pouring_round = 0
        for fid in keyframe_indices:
            kf_type = keyframes[fid]
            if(kf_type=='Open'):
                first_grasp = True
            if kf_type=='Reaching' and first_grasp:
                pouring_round = 1
            end_idx = fid
           
            if kf_type=='Open':
                kf_type = 'Release'
            if kf_type=='Close':
                kf_type = 'Grasping'
            

        all_ts = sorted(self.gp.keys())
        count = 0
        imgs = []       # list of image frames
        frame2ts = []   # corresponding list of video time stamp values in microseconds
        pouring_round = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        first_grasp = False
        kf_type = 'Reaching'
        grasp_time = 0
 

        while success:  
            # print(count)
            # if count>=start and count<=end:
            frame_ts = int((count/fps)*1000000)
            frame2ts.append(frame_ts)

            less = [a for a in self.all_vts if a<=frame_ts]
            idx = len(less)-1

            if idx<len(self.model):
                m,c = self.model[idx]
            else:
                m,c = self.model[len(self.model)-1]
            ts = m*frame_ts + c

            tracker_ts, _ = self.takeClosest(all_ts,ts)

            gaze = self.gp[tracker_ts]
            gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

            if self.visualize:
                img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                hsv = img_hsv[gaze_coords[1]][gaze_coords[0]]
                font = cv2.FONT_HERSHEY_SIMPLEX
                color_name, color_value = get_color_name(hsv)
                
                if(color_name!=''):
                #   print(color_name)
                    cv2.putText(img, color_name, (1430, 250), font, 1.8, color_value, 5, cv2.LINE_AA)

                # print(hsv)
                cv2.putText(img, str(hsv), (230, 250), font, 1.8, (255, 255, 0), 5, cv2.LINE_AA)

                cv2.circle(img,gaze_coords, 25, (255,255,0), 3)
                video.write(img)


            # assign action label for this frame
            if count in keyframe_indices:
                print('keyframe here!')
                idx = keyframe_indices.index(count)
    
                current_recorded_kf = keyframes[keyframe_indices[idx]]
                if idx!=len(keyframe_indices)-1:
                    next_recorded_kf = keyframes[keyframe_indices[idx+1]]

                kf_type = current_recorded_kf  

            if(kf_type=='Pouring'):
                first_grasp = True
            if kf_type=='Reaching' and first_grasp:
                pouring_round = 1
                grasp_time = 0

            if kf_type=='Open':
                kf_type = 'Release' 
            if kf_type=='Close':
                kf_type = 'Grasping'
                grasp_time += 1

            if kf_type=='Grasping' and grasp_time >=1:
                grasp_time+=1


            if grasp_time>=120 and current_recorded_kf=='Close':
                # kf_type = next_recorded_kf
                kf_type = 'Transport'

            if(count%skip==0):
                # 'Other' keyframe is ignored
                if kf_type=='Other':   
                    count+=1
                    success, img = vidcap.read()
                    continue

                # write images
                self.img_count+=1
                img_name = user+'_'+str(trial)+'_'+str(count)+'.jpg'
                cv2.imwrite(self.data_dir+'/KD_images/'+img_name, img)
                f.write(str(self.img_count)+' '+img_name+'\n')
                # seg = self.find_segment(user, count, fps, end_time)
                seg = kf_type
                f2.write(str(self.img_count)+' '+str(self.labels[seg])+'\n')
                # TODO: write normalized gaze coordinates
                f3.write(str(self.img_count)+' '+str(gaze[0])+' '+str(gaze[1])+'\n')
                
                fold_no = int(math.ceil(float(user[2:])/2))
                print('user: '+user+'\tfold_no: '+str(fold_no)+'\tcount: '+str(count))
                fv[fold_no-1].write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

                # this user's data goes to all other training folds
                for i in range(10):
                    if i==fold_no-1:
                        continue
                    ft[i].write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

            count += 1
            success, img = vidcap.read()

    
        if self.visualize:
            cv2.destroyAllWindows()
            video.release()

        f.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()




    def gaze_for_kt_imgs_short(self, seg, user, trial):
        user_dir = self.data_dir + 'videos/' + user + '/' + seg + '/segments/' + str(trial) + '/'
        if not os.path.exists(user_dir):
            return
        bag_dir = '/home/akanksha/Documents/gaze_for_lfd_study_data/gaze_lfd_user_study/' 

        self.read_json(user_dir)
        vidcap = cv2.VideoCapture(user_dir+'fullstream.mp4')
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_time = frame_count/fps
        # print fps     #25 fps
        success, img = vidcap.read()

        if self.visualize:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(user_dir+'gaze_kt_overlayed.avi',fourcc,fps,(1920,1080))

        # open files for image names and labels
        if not os.path.exists(self.data_dir +"KT_50"):
            os.mkdir(self.data_dir +"KT_50")
        if not os.path.exists(self.data_dir +"KT_50/images"):
            os.mkdir(self.data_dir +"KT_50/images")

        f = open(self.data_dir +"KT_50/images.txt",'a')
        f2 = open(self.data_dir +"KT_50/image_class_labels.txt","a")
        f3 = open(self.data_dir +"KT_50/image_gaze.txt","a")
        ft, fv = [], []
        for i in range(10):
            f4 = open(self.data_dir +"KT_50/train"+str(i)+".list","a")
            f5 = open(self.data_dir +"KT_50/val"+str(i)+".list","a")

            ft.append(f4)
            fv.append(f5)


        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)

        bag_file = ''
        
        for file in bagfiles:
            # if (file.endswith("kt-p1.bag")):
            #     bag_file = bagloc + file
            if (file.endswith("kt-p1.bag") and (int(trial)==1 or int(trial)==4)):
                bag_file = bagloc + file
            elif (file.endswith("kt-p2.bag") and (int(trial)==2 or int(trial)==5)):
                bag_file = bagloc + file
            elif (file.endswith("kt-p3.bag") and (int(trial)==3 or int(trial)==6)):
                bag_file = bagloc + file
        
        if bag_file == '':
            print('Bag file does not exist for KT demo, skipping...')
            return
        
        # read bag file
        video_file = user_dir+'fullstream.mp4'
        keyframes, keyframe_indices = self.get_kt_keyframes_labels(video_file, bag_file)

        first_grasp = False
        pouring_round = 0
        for fid in keyframe_indices:
            kf_type = keyframes[fid]
            if(kf_type=='Open'):
                first_grasp = True
            if kf_type=='Reaching' and first_grasp:
                pouring_round = 1
            end_idx = fid
           
            if kf_type=='Open':
                kf_type = 'Release'
            if kf_type=='Close':
                kf_type = 'Grasping'
            

        all_ts = sorted(self.gp.keys())
        count = 0
        imgs = []       # list of image frames
        frame2ts = []   # corresponding list of video time stamp values in microseconds
        pouring_round = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        first_grasp = False
        kf_type = 'Reaching'
        grasp_time = 0
 
        # subample to 52 frames per video
        skip = abs(frame_count/52)
        if user=='KT6' and trial==2:
            skip = abs(frame_count/55)
        if user=='KT5' and trial==3:
            skip = abs(frame_count/55)
        if user=='KT1' and trial==2:
            skip = abs(frame_count/60)
        if user=='KT16' and trial==5:
            skip = abs(frame_count/55)
        if user=='KT14' and trial==1:
            skip = abs(frame_count/57)

        saved = 0
        while success:  
            # print(count)
            # if count>=start and count<=end:
            frame_ts = int((count/fps)*1000000)
            frame2ts.append(frame_ts)

            less = [a for a in self.all_vts if a<=frame_ts]
            idx = len(less)-1

            if idx<len(self.model):
                m,c = self.model[idx]
            else:
                m,c = self.model[len(self.model)-1]
            ts = m*frame_ts + c

            tracker_ts, _ = self.takeClosest(all_ts,ts)

            gaze = self.gp[tracker_ts]
            gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

            if self.visualize:
                img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                hsv = img_hsv[gaze_coords[1]][gaze_coords[0]]
                font = cv2.FONT_HERSHEY_SIMPLEX
                color_name, color_value = get_color_name(hsv)
                
                if(color_name!=''):
                #   print(color_name)
                    cv2.putText(img, color_name, (1430, 250), font, 1.8, color_value, 5, cv2.LINE_AA)

                # print(hsv)
                cv2.putText(img, str(hsv), (230, 250), font, 1.8, (255, 255, 0), 5, cv2.LINE_AA)

                cv2.circle(img,gaze_coords, 25, (255,255,0), 3)
                video.write(img)


            # assign action label for this frame
            if count in keyframe_indices:
                print('keyframe here!')
                idx = keyframe_indices.index(count)
    
                current_recorded_kf = keyframes[keyframe_indices[idx]]
                if idx!=len(keyframe_indices)-1:
                    next_recorded_kf = keyframes[keyframe_indices[idx+1]]

                kf_type = current_recorded_kf  

            if(kf_type=='Pouring'):
                first_grasp = True
            if kf_type=='Reaching' and first_grasp:
                pouring_round = 1
                grasp_time = 0

            if kf_type=='Open':
                kf_type = 'Release' 
            if kf_type=='Close':
                kf_type = 'Grasping'
                grasp_time += 1

            if kf_type=='Grasping' and grasp_time >=1:
                grasp_time+=1


            if grasp_time>=120 and current_recorded_kf=='Close':
                # kf_type = next_recorded_kf
                kf_type = 'Transport'


            if(count%skip==0):
            # if count in subsample:
                # 'Other' keyframe is ignored
                if kf_type=='Other':   
                    count+=1
                    success, img = vidcap.read()
                    continue

                saved+=1

                # write images
                self.img_count+=1
                img_name = user+'_'+str(trial)+'_'+str(count)+'.jpg'
                cv2.imwrite(self.data_dir+'/KT_50/images/'+img_name, img)
                f.write(str(self.img_count)+' '+img_name+'\n')
                # seg = self.find_segment(user, count, fps, end_time)
                seg = kf_type
                f2.write(str(self.img_count)+' '+str(self.labels[seg])+'\n')
                # TODO: write normalized gaze coordinates
                f3.write(str(self.img_count)+' '+str(gaze[0])+' '+str(gaze[1])+'\n')
                
                fold_no = int(math.ceil(float(user[2:])/2))
                # print('frame_count: ',str(frame_count))
                # print(img_name)
                print('user: '+user+'\tfold_no: '+str(fold_no)+'\tcount: '+str(count))
                fv[fold_no-1].write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

                # this user's data goes to all other training folds
                for i in range(10):
                    if i==fold_no-1:
                        continue
                    ft[i].write(img_name+' '+str(self.labels[seg])+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

            count += 1
            success, img = vidcap.read()

    
        if self.visualize:
            cv2.destroyAllWindows()
            video.release()

        f.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()

        if saved<51:
            print('less images than 51!!')
            print(user,trial)
            exit()


    def get_color_name(self,hsv):

        color_ranges = {
            'red':   [[161,140,70],[184,255,255]],
            'green': [[36,64,28],[110,155,220]], #[[36,64,28],[70,155,220]]
            'yellow': [[0,90,100],[32,180,180]],
            'blue': [[94,111,34],[118,165,136]],
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
            'pasta': (0,215,225)
        }

        h,s,v = hsv
        color = ''
        value = None
        for i, (n,r) in enumerate(color_ranges.items()):
            # print(n, r[0][0], r[1][0])
            if h>=r[0][0] and h<=r[1][0]:
                if s>=r[0][1] and s<=r[1][1]:
                    if v>=r[0][2] and v<=r[1][2]:
                        color = n 
                        value = color_val[n]

        pasta_color_range = [[0,30,0],[40,130,100]]
        p = pasta_color_range
        if color=='':
            if h>=p[0][0] and h<=p[1][0]:
                if s>=p[0][1] and s<=p[1][1]:
                    if v>=p[0][2] and v<=p[1][2]:
                        color = 'pasta'
                        value = color_val['pasta']

        return color, value

    def get_kt_keyframes_labels(self, video_file, bag_file):
        vidcap = cv2.VideoCapture(video_file)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, img = vidcap.read()
        print('reading video file')

        keyframes = {}

        last_fixation_color =(0,0,0)
        all_ts = sorted(self.gp.keys())
        count = 0
        imgs = []       # list of image frames
        frame2ts = []   # corresponding list of video time stamp values in microseconds
        videoframe2trackerts = []
        gaze_pts = []

        while success:          
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            frame_ts = int((count/fps)*1000000)
            frame2ts.append(frame_ts)

            less = [a for a in self.all_vts if a<=frame_ts]
            idx = len(less)-1

            if idx<len(self.model):
                m,c = self.model[idx]
            else:
                m,c = self.model[len(self.model)-1]
            ts = m*frame_ts + c
            # print(type(all_ts), type(ts))
            tracker_ts, _ = self.takeClosest(all_ts,int(ts))
            videoframe2trackerts.append(tracker_ts)

            count += 1
            success, img = vidcap.read()

        vidcap.release()
        cv2.destroyAllWindows()

        # find segmentation points on bagfile
        all_keyframe_indices = []
        gripper = {}
        record_k = False
        bag = rosbag.Bag(bag_file)
        print(bag_file)

        # get the start time for KT recording
        start= False
        frame_idx = None

        if bag.get_message_count('/gaze_tracker')!=0:       # gaze_tracker topic was recorded
            for topic, msg, t in bag.read_messages(topics=['/gaze_tracker','/log_KTframe','/joint_states','/vector/right_gripper/stat']):
                #if('vts' in msg.data):
                #print topic
                if (topic=='/log_KTframe'):
                    # print(msg.data)
                    if("Recorded keyframe" in msg.data):
                        record_k = True
                        if 'Reaching' in msg.data:
                            kf_type = 'Reaching'
                        elif 'Grasping' in msg.data:
                            kf_type = 'Grasping'
                        elif 'Transport' in msg.data:
                            kf_type = 'Transport'                       
                        elif 'Pouring' in msg.data:
                            kf_type = 'Pouring'
                        elif 'Return' in msg.data:
                            kf_type = 'Return'
                        elif 'Release' in msg.data:
                            kf_type = 'Release'
                        else:
                            kf_type = 'Other'

                    if("Open" in msg.data):
                        record_k = True
                        kf_type = 'Open'
                    if("Close" in msg.data):
                        record_k = True
                        kf_type = 'Close'

                if (topic == '/gaze_tracker'):
                    if('gp' in msg.data):         
                        gaze_msg = msg.data
                        s = gaze_msg.find('"ts":')
                        e = gaze_msg.find(',')
                        gaze_ts = gaze_msg[s+5:e]
                        tracker_ts, frame_idx = self.takeClosest(videoframe2trackerts,int(gaze_ts))
                        if(record_k == True):  
                            all_keyframe_indices.append(frame_idx)
                            keyframes[frame_idx] = kf_type
                            record_k = False

                if (topic == '/joint_states') and not start and frame_idx!=None:
                    start = True
                    keyframes[frame_idx] = 'Start'

                # if (topic == '/joint_states') and frame_idx!=None:
                    # print('joint states: ',msg)

                # if (topic =='/vector/right_gripper/stat') and frame_idx!=None:
                #     print('gripper stat: ',msg)
                #     # gripper[frame_idx] =  [msg.requested_position, msg.current, msg.position]
                #     gripper[frame_idx] =  [msg.is_ready, msg.is_reset, msg.is_moving]

        
        bag.close()
        return keyframes, all_keyframe_indices


if __name__ == "__main__":
    data = LfdData('pouring', 'v', short_video=False)
    data.create_video_imgs()
    # data.create_kt_imgs()
