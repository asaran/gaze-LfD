# Experiment #4 for the Pouring Task: 
#   Is gaze-based fixation frame different between step KF and non-step KF? 
#       4a. KT demos - ref frame before and after KF (expert users)
#       4b. KT demos - ref frame before and after KF (novice users)

import argparse
from utils 
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import math

parser = argparse.ArgumentParser()
parser.add_argument("-eid", type=str, default="4a", help='Experiment ID')
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
bag_dir = '../../data/gaze_lfd_user_study/'


if args.eid == '4a':
    print('Major reference frame before and after a keyframe')
    print('Measure differences between novice and experts - KT demos')

    expert_target_acc, novice_target_acc = {}, {}
    
    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[expert_target_acc,novice_target_acc]):
        # Get all expert/novice users
        print("processing users' KT Demos...")
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
                    'step': [0, 0],
                    'non-step': [0, 0]
                }


                data, gp, model, all_vts = utils.read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'

                keyframes, keyframe_indices = utils.get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                if(keyframe_indices==[]):
                    continue
                step_kf_indices = utils.get_step_kf_indices(keyframes, keyframe_indices)
                

                hsv_timeline, saccade_indices, fps = utils.get_hsv_color_timeline(data, video_file)


                for fid in keyframe_indices:

                    start_idx = fid - math.floor(fps)
                    if start_idx<0: 
                        start_idx = 0
                    end_idx = fid + math.floor(fps)
                    if end_idx > len(hsv_timeline):
                        end_idx = len(hsv_timeline) - 1
                    fixations_before = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, fid)
                    fixations_after = utils.filter_fixations(video_file, model, gp, all_vts, demo_type, saccade_indices, fid, end_idx)


                    if fid in step_kf_indices:
                        kf_type = 'step'
                    else:
                        kf_type = 'non-step'

                    # assign max value to the color of the default target of this KF
                    max_val_before = 0
                    max_val_after = 0

                    for key, val in fixations_before.items():
                        if(val==1):
                            print('continuing')
                            continue
                        if val>max_val_before:
                            max_val_before = val
                            max_color_before = key


                    for key, val in fixations_after.items():
                        if(val==-1):
                            continue
                        if val>max_val_after:
                            max_val_after = val
                            max_color_after = key

                    if(max_val_before>0 and max_val_after>0):
                        print(kf_type, keyframes[fid])
                        print('****max colors:')
                        print(max_color_before, max_color_after)
                        if max_color_after!=max_color_before and max_color_before!='other' and max_color_after!='other':
                            target_acc[kf_type][0] += 1
                        target_acc[kf_type][1] += 1

                all_fix[user[2:]] = target_acc
                print(target_acc)

    with open('4a_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        kf_names_ = expert_target_acc[experts[0][2:]].keys()
        kf_names = [kf_names_[0], kf_names_[0]+ ' total', kf_names_[1], kf_names_[1]+ ' total']
        u_kf_names = ['User ID'] + kf_names
        expert_writer.writerow(u_kf_names)
        for u, acc in expert_target_acc.items():
            value_list = [acc[i][j] for i in kf_names_ for j in [0,1]]
            value_list = [u] + value_list
            expert_writer.writerow(value_list)

    with open('4a_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        kf_names_ = novice_target_acc[novices[0][2:]].keys()
        kf_names = [kf_names_[0], kf_names_[0]+ ' total', kf_names_[1], kf_names_[1]+ ' total']
        u_kf_names = ['User ID'] + kf_names
        novice_writer.writerow(u_kf_names)
        for u, acc in novice_target_acc.items():
            value_list = [acc[i][j] for i in kf_names_ for j in [0,1]]
            value_list = [u] + value_list
            novice_writer.writerow(value_list)   




if args.eid == '4b':
    print('Major reference frame before and after a keyframe')
    print('Measure importance of gaze as a feature with an ROC curve - KT demos')
    
    expert_target_acc, novice_target_acc = {}, {}

    for u, user_dir, all_fix in zip([experts,novices],[expert_dir,novice_dir],[expert_target_acc,novice_target_acc]):

        # Get all expert/novice users
        print("processing users' KT Demos...")
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

                if(int(seg)!=1 and int(seg)!=4):
                    continue

                if demo_type!='k':
                    continue

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

                target_count = {
                    'step': [],         # a list of target frame counts before and after per keyframe
                    'non-step': []
                }

                data, gp, model, all_vts = utils.read_json(a+seg)
                video_file = a+seg+'/fullstream.mp4'

                keyframes, keyframe_indices = utils.get_kt_keyframes_labels(all_vts, model, gp, video_file, bag_file)
                if(keyframe_indices==[]):
                    continue
                step_kf_indices = utils.get_step_kf_indices(keyframes, keyframe_indices)
                

                hsv_timeline, saccade_indices, fps = utils.get_hsv_color_timeline(data, video_file)


                for fid in keyframe_indices:

                    start_idx = fid - 3*math.floor(fps)
                    if start_idx<0: 
                        start_idx = 0
                    end_idx = fid + 3*math.floor(fps)
                    if end_idx > len(hsv_timeline):
                        end_idx = len(hsv_timeline) - 1
                    fixations_before = utils.filter_fixation_counts(video_file, model, gp, all_vts, demo_type, saccade_indices, start_idx, fid)
                    fixations_after = utils.filter_fixation_counts(video_file, model, gp, all_vts, demo_type, saccade_indices, fid, end_idx)

                    colored_frames_before = 0
                    colored_frames_after = 0
                    for key, val in fixations_before.items():
                        if key=='red' or key=='blue' or key=='green' or key=='yellow' or key=='pasta':
                            colored_frames_before+=val

                    for key, val in fixations_after.items():
                        if key=='red' or key=='blue' or key=='green' or key=='yellow' or key=='pasta':
                            colored_frames_after+=val

                    if fid in step_kf_indices:
                        kf_type = 'step'
                    else:
                        kf_type = 'non-step'

                    # assign max value to the color of the default target of this KF
                    max_val_before = 0
                    max_val_after = 0

                    for key, val in fixations_before.items():
                        if(val==-1):
                            print('continuing')
                            continue
                        if colored_frames_before>0:
                            if val*100.0/colored_frames_before>max_val_before:
                                max_val_before = val*100.0/colored_frames_before
                                max_color_before = key


                    for key, val in fixations_after.items():
                        if(val==-1):
                            continue
                        if colored_frames_after>0:
                            if val*100.0/colored_frames_after>max_val_after:
                                max_val_after = val*100.0/colored_frames_after
                                max_color_after = key

                    if(max_val_before>0 and max_val_after>0):
                        val_color_after = fixations_after[max_color_before]
                        target_count[kf_type].append(abs(max_val_before-val_color_after))
                        

                all_fix[user[2:]] = target_count
                print(target_count)

    thresholds = range(0,105,5)

    # CSV files 
    with open('4b_kt_expert.csv', mode='w') as expert_file:
        expert_writer = csv.writer(expert_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['Threshold', 'True Positive Rate', 'False Positive Rate']
        expert_writer.writerow(header)
        
        for t in thresholds:
            tp_num, tp_den, fp_num, fp_den  = 0, 0, 0, 0
            for u, diff in expert_target_acc.items():
                step_correct = float(sum(i > t for i in diff['step']))
                non_step_correct = float(sum(i < t for i in diff['non-step']))
                tp_num += step_correct
                tp_den += len(diff['step'])
                fp_num += non_step_correct
                fp_den += len(diff['non-step'])
            tp = tp_num/tp_den
            fp = fp_num/fp_den
            expert_writer.writerow([t, tp, fp])

    with open('4b_kt_novice.csv', mode='w') as novice_file:
        novice_writer = csv.writer(novice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['Threshold', 'True Positive Rate', 'False Positive Rate']
        novice_writer.writerow(header)
        for t in thresholds:
            tp_num, tp_den, fp_num, fp_den  = 0, 0, 0, 0
            for u, diff in novice_target_acc.items():
                step_correct = float(sum(i > t for i in diff['step']))
                non_step_correct = float(sum(i < t for i in diff['non-step']))
                tp_num += step_correct
                tp_den += len(diff['step'])
                fp_num += non_step_correct
                fp_den += len(diff['non-step'])
            tp = tp_num/tp_den
            fp = fp_num/fp_den
            novice_writer.writerow([t, tp, fp])
