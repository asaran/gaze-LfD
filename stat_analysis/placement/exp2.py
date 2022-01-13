# Experiments for the Placement Task
# 2. Gaze fixations on bowl and plate for different relative placement strategy
#       a. Plate versus Bowl (Video demo for all users)
#       b. Plate versus Bowl (KT demo for expert users)
#       c. Plate versus Bowl (KT demo for novice users)


import argparse
from utils import takeClosest, get_hsv_color_timeline, get_color_name_from_hist
from utils import get_video_keyframes, read_json, filter_fixations, get_kt_keyframes
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=str, default="2a", help='Experiment ID')
args = parser.parse_args()

expert_dir = '../../data/reward/experts/'    
experts = os.listdir(expert_dir)
# print(users)

novice_dir = '../../data/reward/novices/'    
novices = os.listdir(novice_dir)

all_dir = '../../data/reward/all_users/'
all_users = os.listdir(all_dir)

order = {'KT1':'kvpb','KT2':'kvbp','KT3':'vkpb','KT4':'vkbp','KT5':'kvbp','KT6':'kvpb','KT7':'vkbp','KT8':'vkpb','KT9':'kvpb','KT10':'kvbp',\
        'KT11':'vkpb','KT12':'vkbp','KT13':'kvbp','KT14':'kvpb','KT15':'vkbp','KT16':'vkpb','KT17':'kvpb','KT18':'vkbp','KT19':'vkpb','KT20':'vkbp'}


condition_names = {
    'k': 'KT demo',
    'v': 'Video demo',
    'p': 'plate target',
    'b': 'bowl target'
}

video_kf_file = 'video_kf.txt'
bag_dir = '/home/akanksha/Documents/gaze_for_lfd_study_data/gaze_lfd_user_study/'

if args.eid == '2a':
    print('Gaze fixations on bowl and plate for different relative placement strategy')
    print('Plate versus Bowl (Video demo for all users)')

    user_dir = all_dir
    bowl_fixations, plate_fixations = {}, {}
    for i in range(len(all_users)):
        user = all_users[i]
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=2 else exps[1]
            cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
            exp_id = demo_type + cond

            if demo_type!='v':
                continue

            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
            keyframe_indices = get_video_keyframes(user, int(seg), video_file, video_kf_file)
            print(keyframe_indices)
            start_idx, end_idx = keyframe_indices['Start'][0], keyframe_indices['Stop'][0]
            fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

            if(cond=='p'):
                plate_fixations[user[2:]] = fixations
            if (cond=='b'):
                bowl_fixations[user[2:]] = fixations

    with open('2a_video_plate_all.csv', mode='w') as plate_file:
        plate_writer = csv.writer(plate_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = plate_fixations[all_users[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        plate_writer.writerow(u_color_names)
        for us,fix in plate_fixations.items():
            value_list = [fix[i] for i in color_names]
            value_list = [us] + value_list
            plate_writer.writerow(value_list)

    with open('2a_video_bowl_all.csv', mode='w') as bowl_file:
        bowl_writer = csv.writer(bowl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = bowl_fixations[all_users[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        bowl_writer.writerow(u_color_names)
        for us,fix in bowl_fixations.items():
            value_list = [fix[i] for i in color_names]
            value_list = [us] + value_list
            bowl_writer.writerow(value_list)
    

if args.eid == '2b':
    print('Perecentage accuarcy to predict instruction from gaze')
    print('Plate versus Bowl (KT demo for expert users)')
    
    user_dir = expert_dir
    bowl_fixations, plate_fixations = {}, {}
    for i in range(len(experts)):
        user = experts[i]
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)


        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=2 else exps[1]
            cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
            exp_id = demo_type + cond


            if demo_type!='k':
                continue

            bag_file = ''
            for file in bagfiles:
                if (file.endswith("kt-irl-bowl.bag") and demo_type=='k' and cond=='b'):
                    bag_file = bagloc + file
                elif(file.endswith("kt-irl-plate.bag") and demo_type=='k' and cond=='p'):
                    bag_file = bagloc + file
            
            if bag_file == '':
                print('Bag file does not exist for KT demo, skipping...')
                continue

            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
            print(keyframes)
            start_idx, end_idx = keyframes[0], keyframes[-1]
            hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
            fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

            if(cond=='p'):
                plate_fixations[user[2:]] = fixations
            if (cond=='b'):
                bowl_fixations[user[2:]] = fixations

    with open('2b_KT_plate_experts.csv', mode='w') as plate_file:
        plate_writer = csv.writer(plate_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = plate_fixations[experts[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        plate_writer.writerow(u_color_names)
        for us,fix in plate_fixations.items():
            value_list = [fix[i] for i in color_names]
            value_list = [us] + value_list
            plate_writer.writerow(value_list)

    with open('2b_KT_bowl_experts.csv', mode='w') as bowl_file:
        bowl_writer = csv.writer(bowl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = bowl_fixations[experts[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        bowl_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us,fix in bowl_fixations.items():
            value_list = [fix[i] for i in color_names]
            value_list = [us] + value_list
            bowl_writer.writerow(value_list)


if args.eid == '2c':
    print('Perecentage accuarcy to predict instruction from gaze')
    print('Plate versus Bowl (KT demo for novice users)')

    user_dir = novice_dir
    bowl_fixations, plate_fixations = {}, {}
    for i in range(len(novices)):
        user = novices[i]
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)


        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=2 else exps[1]
            cond = exps[2] if (int(seg)==1 or int(seg)==3) else exps[3]
            exp_id = demo_type + cond


            if demo_type!='k':
                continue

            bag_file = ''
            for file in bagfiles:
                if (file.endswith("kt-irl-bowl.bag") and demo_type=='k' and cond=='b'):
                    bag_file = bagloc + file
                elif(file.endswith("kt-irl-plate.bag") and demo_type=='k' and cond=='p'):
                    bag_file = bagloc + file
            
            if bag_file == '':
                print('Bag file does not exist for KT demo, skipping...')
                continue

            data, gp, model, all_vts = read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            keyframes = get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
            print(keyframes)
            start_idx, end_idx = keyframes[0], keyframes[-1]
            hsv_timeline, saccade_indices = get_hsv_color_timeline(data, video_file)
            fixations = filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

            if(cond=='p'):
                plate_fixations[user[2:]] = fixations
            if (cond=='b'):
                bowl_fixations[user[2:]] = fixations

    with open('2c_KT_plate_novice.csv', mode='w') as plate_file:
        plate_writer = csv.writer(plate_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = plate_fixations[novices[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        plate_writer.writerow(u_color_names)
        for us,fix in plate_fixations.items():
            value_list = [fix[i] for i in color_names]
            value_list = [us] + value_list
            plate_writer.writerow(value_list)

    with open('2c_KT_bowl_novice.csv', mode='w') as bowl_file:
        bowl_writer = csv.writer(bowl_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = bowl_fixations[novices[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        bowl_writer.writerow(u_color_names)
        for us,fix in bowl_fixations.items():
            value_list = [fix[i] for i in color_names]
            value_list = [us] + value_list
            bowl_writer.writerow(value_list)