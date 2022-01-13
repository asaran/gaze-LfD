#!/usr/bin/env python2
'''This code run gaze augmented BIRL with dummy data based on the 
instruction type (relative placement preference, i.e. w.r.t bowl or plate) and demo type'''

import random
import numpy as np
import birl
import utils
import matplotlib.pyplot as plt
import complexreward as cr
import sys
import argparse
from copy import deepcopy
import pickle as pkl

def test_placements(true_reward, num_test):
    test_rbfs = []
    for i in range(num_test):
        #generate new centers, but keep weights
        num_objs = true_reward.num_objects
        new_centers = np.random.rand(num_objs, 2)
        obj_weights = true_reward.obj_weights.copy()
        abs_weights = true_reward.abs_weights.copy()
        new_rbf = utils.RbfComplexReward(new_centers, obj_weights, abs_weights)
        test_rbfs.append(new_rbf)
    return test_rbfs


#return mean and standard deviation of loss over test placements
def calc_test_reward_loss(test_placements, map_params, visualize=False):
    losses = []
    for placement in test_placements:
        test_config = placement.obj_centers
        true_params = (placement.obj_weights, placement.abs_weights)
        ploss = cr.calculate_policy_loss(test_config, true_params, map_params)
        if visualize:
            test_map = utils.RbfComplexReward(test_config, map_params[0], map_params[1])
            utils.visualize_reward(test_map, "testing with map reward")
            plt.show()
        losses.append(ploss)
    losses = np.array(losses)
    return np.mean(losses), np.std(losses), np.max(losses)


def calc_test_placement_loss(test_placements, map_params, visualize=False):
    losses = []
    cnt = 0
    for placement in test_placements:
        #print cnt
        cnt += 1
        test_config = placement.obj_centers
        true_params = (placement.obj_weights, placement.abs_weights)
        ploss = cr.calculate_placement_loss(test_config, true_params, map_params)
        if visualize:
            test_map = utils.RbfComplexReward(test_config, map_params[0], map_params[1])
            utils.visualize_reward(test_map, "testing with map reward")
            plt.show()
        losses.append(ploss)
    losses = np.array(losses)
    return np.mean(losses), np.std(losses), np.max(losses)


if __name__=="__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--distractors', '-d', action='store_true',
            help='distractor objects considered or not')
    parser.add_argument('--use_gaze', '-g', action='store_true',
            help='use gaze augmented BIRL or standard BIRL')
    parser.add_argument('--exp', default='bowl', type=str,
            help='choose bowl | plate (default: bowl). Signifies experiment instruction was with respect to bowl or plate.')
    parser.add_argument('--demo_type', default='video', type=str,
            help='choose video | KT (default: video). Signifies demonstration type.')
    parser.add_argument('--num_test', default=100, type=int, 
        help='number of test placements to consider in simulation')
    parser.add_argument('--seed', default=12345, type=int, 
        help='random seed')
    args = parser.parse_args()


    rand_seed = args.seed
    np.random.seed(rand_seed)
    num_test = args.num_test
    exp = args.exp # 'plate' or 'bowl'
    demo_type = args.demo_type # 'video' or 'KT'
    distractors = args.distractors
    use_gaze = args.use_gaze
    

    beta=100.0
    num_steps = 1000
    step_std = 0.05
    burn = 0
    skip = 25

    if distractors:
        num_objects = 4 
    else:
        num_objects = 2 # plate and bowl

    #object weights are for center, top left, top right, bottom left, bottom right
    obj2_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #distractor
    obj3_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #distractor
    demo = {}
    
    if(demo_type=='video'):
        if exp=='bowl':
            data_file = 'data/video_bowl.txt'
            obj1_weights = np.array([0.0, 0.5, 0.0, 0.5, 0.0]) #bowl: equal weight on top left and bottom left rbf results in placement directly to left of object
            obj0_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #plate


        elif exp=='plate':
            data_file = 'data/video_plate.txt'
            obj1_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #bowl: equal weight on top left and bottom left rbf results in placement directly to left of object
            obj0_weights = np.array([0.0, 0.0, 0.5, 0.0, 0.5]) #plate


    if(demo_type=='KT'):
        if exp=='bowl':
            data_file = 'data/KT_bowl_distractors.txt'
            obj1_weights = np.array([0.0, 0.0, 0.5, 0.0, 0.5]) #bowl: equal weight on top left and bottom left rbf results in placement directly to left of object
            obj0_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #plate

        
        elif exp=='plate':
            data_file = 'data/KT_plate_distractors.txt'
            obj1_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) #bowl
            obj0_weights = np.array([0.0, 0.5, 0.0, 0.5, 0.0]) #plate: equal weight on top left and bottom left rbf results in placement directly to left of object


    f = open(data_file, 'r')
    data = f.read()
    data = data.strip().split('\n')
    num_demos = len(data)-1

    objs = data[0].split('\t') #object labels
    for obj in objs:
        demo[obj]=[] 
    for i in range(1,len(data)):
        line = data[i]
        line = line.split('\t')
        line = [x.split(',') for x in line]
        for j in range(len(line)):
            line[j] = [float(a) for a in line[j]]
        for j in range(len(objs)):
            demo[objs[j]].append(line[j]) 

    if distractors:
        assert('orange' in objs)
        assert('purple' in objs)
        obj_weights = np.concatenate((obj0_weights, obj1_weights, obj2_weights, obj3_weights))
    else:
        obj_weights = np.concatenate((obj0_weights, obj1_weights))
    abs_weights = np.array([0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]) # no absolute placement preferences
    
    num_obj_weights = len(obj_weights)
    num_abs_weights = len(abs_weights)

    birl = birl.BIRL(num_obj_weights, num_abs_weights, beta, num_steps, step_std, burn, skip)

    for i in range(num_demos):
        #generate random object placements
        if distractors:
            obj_centers = np.array([demo['plate'][i],demo['bowl'][i],demo['orange'][i],demo['purple'][i]])
        else:
            obj_centers = np.array([demo['plate'][i],demo['bowl'][i]])
        look_times = np.array(demo['gaze'][i])
        best_x = np.array(demo['ladle'][i]) 
        true_rbf = utils.RbfComplexReward(obj_centers, obj_weights, abs_weights)

        birl.add_gaze_demonstration(obj_centers, best_x, look_times)
        utils.visualize_reward(true_rbf, "demo {} and ground truth reward".format(i) )

    #run birl to get MAP estimate
    birl.run_inference(gaze=use_gaze)

    # get map reward weights
    map_obj_wts, map_abs_wts = birl.get_map_params()

    # test placements in simulation for 100 configurations 
    test_rbfs = test_placements(true_rbf, num_test)
    mean_obj_wts, mean_abs_wts = birl.get_map_params()
    # print "obj weights", mean_obj_wts
    # print "abs weights", mean_abs_wts
    ave_loss, std_loss, max_loss = calc_test_reward_loss(test_rbfs, birl.get_map_params(), False)
    print "policy loss:", ave_loss, std_loss, max_loss

    ave_loss, std_loss, max_loss = calc_test_placement_loss(test_rbfs, birl.get_map_params(), False)
    print "placement loss:", ave_loss, std_loss, max_loss
    print "reward diff:", np.linalg.norm(true_rbf.obj_weights - mean_obj_wts), np.linalg.norm(true_rbf.abs_weights - mean_abs_wts)