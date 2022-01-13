# Experiment #2 for the Pouring Task: 
#    Perecentage accuarcy to predict reference frame per keyframe 
#       2a. Video versus KT demos (expert users)
#       2b. Video versus KT demos (novice users)     


import argparse
from utils 
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math

parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=str, default="2a", help='Experiment ID')
args = parser.parse_args()

expert_dir = '../../data/pouring/experts/'    
experts = os.listdir(expert_dir)
# print(users)

novice_dir = '../../data/pouring/novices/'    
novices = os.listdir(novice_dir)

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

condition_names = {
    'k': 'KT demo',
    'v': 'Video demo'
}

video_kf_file = 'video_kf.txt'
bag_dir = '../../data/gaze_lfd_user_study/'


if args.eid == '2a':
    print('Perecentage accuarcy to predict reference frame per keyframe')
    print('Video versus KT demos (expert users)')

    kt_target_acc, video_target_acc = {}, {}

    target_objects = {
        'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
        'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
        'Transport': [['red'], ['blue']],
        'Pouring': [['red', 'pasta'],['blue','pasta']],
        'Return': [['other'],['other']],
        'Release': [['green'],['yellow']]
    }
    user_dir = expert_dir
    print("processing Expert Users' Video Demos...")
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
            demo_type = exps[0] if int(seg)<=3 else exps[1]

            bag_file = ''
            if(demo_type=='k'):
                for file in bagfiles:
                    if (file.endswith("kt-p1.bag") and (int(seg)==1 or int(seg)==4)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p2.bag") and (int(seg)==2 or int(seg)==5)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p3.bag") and (int(seg)==3 or int(seg)==6)):
                        bag_file = bagloc + file
                
                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

            target_acc = {
                'Reaching': [0, 0],
                'Grasping': [0, 0],
                'Transport': [0, 0],
                'Pouring': [0, 0],
                'Return': [0, 0],
                'Release': [0, 0]
            }

            data, gp, model, all_vts = utils.read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            if(demo_type=='k'):
                keyframes, keyframe_indices = utils.get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
            if(demo_type=='v'):
                keyframes, keyframe_indices = utils.get_video_keyframe_labels(user, video_file, video_kf_file)

            # Find end of first pouring - start of next pouring
            first_grasp = False
            pouring_round = 0

            hsv_timeline, saccade_indices, _ = utils.get_hsv_color_timeline(data, video_file)

            if(demo_type=='k'):
                start_idx = 0   
                for fid in keyframe_indices:
                    kf_type = keyframes[fid]
                    if(kf_type=='Open'):
                        first_grasp = True
                    if kf_type=='Reaching' and first_grasp:
                        pouring_round = 1
                    end_idx = fid
                    fixations = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
                    

                    if kf_type=='Open' or kf_type=='Close':
                        kf_type = 'Grasping'
                    if kf_type not in target_objects.keys():
                        start_idx = end_idx
                        continue

                    # assign max value to the color of the default target of this KF
                    max_val = 0
                    for o in target_objects[kf_type][pouring_round]:
                        if(fixations[o]!=-1):
                            max_val += fixations[o]
                    max_color = target_objects[kf_type][pouring_round][0]
                    for key, val in fixations.items():
                        if(val==-1): #bug
                            continue
                        if val>max_val:
                            max_val = val
                            max_color = key
                    if(max_val>0):
                        if max_color == target_objects[kf_type][pouring_round][0]:
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1

                    # One plot showing both novice and expert numbers for objects, other
                    start_idx = end_idx
                kt_target_acc[user[2:]+'_'+str(seg)] = target_acc

            if(demo_type=='v'): 
                for j in range(1,len(keyframe_indices)-1):
                    fid = keyframe_indices[j]
                    kf_type = keyframes[fid]
                    if(kf_type=='Pouring'):
                        first_grasp = True
                    if kf_type=='Reaching' and first_grasp:
                        pouring_round = 1
                    start_idx = fid
                    end_idx = keyframe_indices[j+1]
                    fixations = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                    # assign max value to the color of the default target of this KF
                    max_val = 0
                    for o in target_objects[kf_type][pouring_round]:
                        if(fixations[o]!=-1):
                            max_val += fixations[o]
                    max_color = target_objects[kf_type][pouring_round][0]
                    for key, val in fixations.items():
                        if(val==-1):
                            continue
                        if val>max_val:
                            max_val = val
                            max_color = key
                    if(max_val>0):
                        if max_color == target_objects[kf_type][pouring_round][0]:
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1

                video_target_acc[user[2:]+'_'+str(seg)] = target_acc

    with open('2a_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        kf_names = kt_target_acc[kt_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)

        for u,acc in kt_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1  for i in kf_names]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)

    with open('2a_video_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        kf_names = video_target_acc[video_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        for u,acc in video_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1] if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)
    

if args.eid == '2b':
    print('Perecentage accuarcy to predict reference frame per keyframe')
    print('Video versus KT demos (novice users)')
    
    kt_target_acc, video_target_acc = {}, {}

    target_objects = {
        'Reaching': [['green', 'pasta'], ['yellow', 'pasta']],
        'Grasping': [['green', 'pasta'], ['yellow', 'pasta']],
        'Transport': [['red'], ['blue']],
        'Pouring': [['red', 'pasta'],['blue','pasta']],
        'Return': [['other'],['other']],
        'Release': [['green'],['yellow']]
    }
    user_dir = novice_dir
    print("processing Expert Users' Video Demos...")
    for i in range(len(novices)):
        user = novices[i]
        print(user) 
        dir_name = os.listdir(user_dir+user)

        a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
        d = os.listdir(a)

        exps = order[user]

        bagloc = bag_dir + user + '/bags/'
        bagfiles = os.listdir(bagloc)
        
        for seg in d:
            print('Segment ', seg)
            demo_type = exps[0] if int(seg)<=3 else exps[1]


            bag_file = ''
            if(demo_type=='k'):
                for file in bagfiles:
                    if (file.endswith("kt-p1.bag") and (int(seg)==1 or int(seg)==4)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p2.bag") and (int(seg)==2 or int(seg)==5)):
                        bag_file = bagloc + file
                    elif (file.endswith("kt-p3.bag") and (int(seg)==3 or int(seg)==6)):
                        bag_file = bagloc + file
                

                if bag_file == '':
                    print('Bag file does not exist for KT demo, skipping...')
                    continue

            target_acc = {
                'Reaching': [0, 0],
                'Grasping': [0, 0],
                'Transport': [0, 0],
                'Pouring': [0, 0],
                'Return': [0, 0],
                'Release': [0, 0]
            }

            data, gp, model, all_vts = utils.read_json(a+seg)
            video_file = a+seg+'/fullstream.mp4'
            if(demo_type=='k'):
                keyframes, keyframe_indices = utils.get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
            if(demo_type=='v'):
                keyframes, keyframe_indices = utils.get_video_keyframe_labels(user, video_file, video_kf_file)
            first_grasp = False
            pouring_round = 0

            hsv_timeline, saccade_indices, _ = utils.get_hsv_color_timeline(data, video_file)

            if(demo_type=='k'):
                start_idx = 0
                for fid in keyframe_indices:
                    kf_type = keyframes[fid]
                    if(kf_type=='Open'):
                        first_grasp = True
                    if kf_type=='Reaching' and first_grasp:
                        pouring_round = 1
                    end_idx = fid
                    fixations = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
                    

                    if kf_type=='Open' or kf_type=='Close':
                        kf_type = 'Grasping'
                    if kf_type not in target_objects.keys():
                        start_idx = end_idx
                        continue

                    # assign max value to the color of the default target of this KF
                    max_val = 0
                    for o in target_objects[kf_type][pouring_round]:
                        if(fixations[o]!=-1):
                            max_val += fixations[o]

                    max_color = target_objects[kf_type][pouring_round][0]
                    for key, val in fixations.items():
                        if(val==-1):
                            continue
                        if val>max_val:
                            max_val = val
                            max_color = key

                    if(max_val>0):
                        if max_color == target_objects[kf_type][pouring_round][0]:
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1

                    start_idx = end_idx
                kt_target_acc[user[2:]+'_'+str(seg)] = target_acc

            if(demo_type == 'v'):
                for j in range(1,len(keyframe_indices)-1):
                    fid = keyframe_indices[j]
                    kf_type = keyframes[fid]
                    if(kf_type=='Pouring'):
                        first_grasp = True
                    if kf_type=='Reaching' and first_grasp:
                        pouring_round = 1
                    start_idx = fid
                    end_idx = keyframe_indices[j+1]
                    fixations = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)

                    # assign max value to the color of the default target of this KF
                    max_val = 0
                    for o in target_objects[kf_type][pouring_round]:
                        if(fixations[o]!=-1):
                            max_val += fixations[o]
                    max_color = target_objects[kf_type][pouring_round][0]
                    for key, val in fixations.items():
                        if(val==-1):
                            continue
                        if val>max_val:
                            max_val = val
                            max_color = key
                    if(max_val>0):
                        if max_color == target_objects[kf_type][pouring_round][0]:
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1

                video_target_acc[user[2:]+'_'+str(seg)] = target_acc


    with open('2b_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        kf_names = kt_target_acc[kt_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        for u, acc in kt_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1 for i in kf_names]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)

    with open('2b_video_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        kf_names = video_target_acc[video_target_acc.keys()[0]].keys()
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        for u, acc in video_target_acc.items():
            value_list = [acc[i][0]*100.0/acc[i][1]  if acc[i][1]!=0 else -1  for i in kf_names]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)