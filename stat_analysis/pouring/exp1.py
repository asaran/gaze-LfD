# Experiment #1 for the Pouring Task
#    Percentage of time during entire demo spent fixating on objects, gripper or other parts of workspace 
#       a. Measure differences between novice and experts - video demos
#       b. Measure differences between novice and experts - KT demos


import argparse
from utils 
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math

parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=str, default="1a", help='Experiment ID')
args = parser.parse_args()

expert_dir = '../../data/pouring/experts/'    
experts = os.listdir(expert_dir)

novice_dir = '../../data/pouring/novices/'    
novices = os.listdir(novice_dir)

order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
        'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

condition_names = {
    'k': 'KT demo',
    'v': 'Video demo'
}

video_kf_file = 'video_kf.txt'
bag_dir = '/home/user/Documents/gaze_for_lfd_study_data/gaze_lfd_user_study/'

if args.eid == '1a':
    print('Percentage of time during entire demo - spent on objects or other parts of workspace')
    print('Measure differences between novice and experts - video demos')

    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' Video Demos...")
        for i in range(len(u)):
            user = u[i]
            print(user) 
            dir_name = os.listdir(user_dir+user)

            a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
            d = os.listdir(a)

            exps = order[user]

            for seg in d:
                print('Segment ', seg)
                demo_type = exps[0] if int(seg)<=3 else exps[1]

                if demo_type!='v':
                    continue

                data, gp, model, all_vts = utils.read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                hsv_timeline, saccade_indices, _ = utils.get_hsv_color_timeline(data, video_file)
                keyframe_indices = utils.get_video_keyframes(user, video_file, video_kf_file)
                start_idx, end_idx = keyframe_indices['Start'][0], keyframe_indices['Stop'][0]
                fixations = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, end_idx)
                all_fix[user[2:]+'_'+str(seg)] = fixations

    with open('1a_video_expert_3trials.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[experts[0][2:]+'_1'].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('1a_video_novice_3trials.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[novices[0][2:]+'_1'].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        # no_of_colors = length(color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)

if args.eid == '1b':
    print('Percentage of time during entire demo - spent on objects, gripper or other parts of workspace')
    print('Measure differences between novice and experts - KT demos')
    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' Video Demos...")
        for i in range(len(u)):
            user = u[i]
            print(user) #KT1,KT2
            dir_name = os.listdir(user_dir+user)

            a = user_dir+user+'/'+dir_name[0]+'/'+'segments/'
            d = os.listdir(a)

            exps = order[user]

            bagloc = bag_dir + user + '/bags/'
            bagfiles = os.listdir(bagloc)


            for seg in d:
                print('Segment ', seg)
                demo_type = exps[0] if int(seg)<=3 else exps[1]

                if demo_type!='k':
                    continue

                bag_file = ''
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

                data, gp, model, all_vts = utils.read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'
                keyframes = utils.get_kt_keyframes(all_vts, model, gp, video_file, bag_file)
                start_idx, end_idx = keyframes[0], keyframes[-1]
                hsv_timeline, saccade_indices, _ = utils.get_hsv_color_timeline(data, video_file)

                fixations = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx,end_idx)
                all_fix[user[2:]+'_'+str(seg)] = fixations

    with open('1b_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[all_expert_fix.keys()[0]].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('1b_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[all_novice_fix.keys()[0]].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)