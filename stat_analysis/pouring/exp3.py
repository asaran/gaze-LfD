#  Experiment #3 for the Pouring Task: 
#     Do keyframe KT and video demos match in overall fixations?
#       3a. Novice versus expert users (video demos)
#       3b. Novice versus expert users (KT demos)

import argparse
from utils 
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math

parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=str, default="3a", help='Experiment ID')
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

video_kf_file = 'video_kf.txt'  # manually annotated video keyframe timestamps
bag_dir = '../../data/gaze_lfd_user_study/' # bag files for the user study data


if args.eid == '3a':
    print('Measure differences between novice and experts - video demos on task objects only')
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
                keyframes, all_keyframe_indices = utils.get_video_keyframe_labels(user, video_file, video_kf_file)
                if(keyframes==[]):
                    continue
                hsv_timeline, saccade_indices, _ = utils.get_hsv_color_timeline(data, video_file)

                fixations = utils.filter_fixations_ignore_black(video_file, model, gp, all_vts, demo_type, saccade_indices, all_keyframe_indices, keyframes)
                all_fix[user[2:]] = fixations

    with open('3a_video_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[experts[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('3a_video_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[novices[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)


if args.eid == '3b':
    print('Measure differences between novice and experts - KT demos')
    all_expert_fix, all_novice_fix = {}, {}
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[all_expert_fix,all_novice_fix]):

        # Get all expert users
        print("processing Expert Users' KT Demos...")
        for i in range(len(u)):
            user = u[i]
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
                keyframes, all_keyframe_indices = utils.get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                if(keyframes==[]):
                    continue
                hsv_timeline, saccade_indices, _ = utils.get_hsv_color_timeline(data, video_file)

                fixations = utils.filter_fixations_ignore_black(video_file, model, gp, all_vts, demo_type, saccade_indices, all_keyframe_indices, keyframes)
                all_fix[user[2:]] = fixations

    with open('3b_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_expert_fix[experts[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        expert_writer.writerow(u_color_names)
        for us, expert_fix in all_expert_fix.items():
            value_list = [expert_fix[i] for i in color_names]
            value_list = [us] + value_list
            expert_writer.writerow(value_list)

    with open('3b_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_names = all_novice_fix[novices[0][2:]].keys()
        u_color_names = ['User ID'] + color_names
        novice_writer.writerow(u_color_names)
        for us, novice_fix in all_novice_fix.items():
            value_list = [novice_fix[i] for i in color_names]
            value_list = [us] + value_list
            novice_writer.writerow(value_list)