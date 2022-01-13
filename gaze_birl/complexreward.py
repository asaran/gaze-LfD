# -*- coding: utf-8 -*-

import birl
import utils
import numpy as np
import matplotlib.pyplot as plt


#calculate the policy loss between the hypothesis return and the map return
def calculate_policy_loss(config, hyp_params, map_params):
    #calculate reward for optimal placement under hyp_reward
    hyp_obj_weights, hyp_abs_weights = hyp_params
    hyp_reward_fn = utils.RbfComplexReward(config, hyp_obj_weights, hyp_abs_weights)
    #get optimal placement under the hypothesis reward function and new configuration
    hyp_placement, hyp_return = hyp_reward_fn.estimate_best_placement()

    #calculate reward for map placement under hyp_reward
    map_obj_weights, map_abs_weights = map_params
    map_reward_fn = utils.RbfComplexReward(config, map_obj_weights, map_abs_weights)
    #get optimal placement under map reward function and new configuration
    map_placement, _ = map_reward_fn.estimate_best_placement()
    map_return = hyp_reward_fn.get_reward(map_placement)

    return hyp_return - map_return

def calculate_placement_loss(config, hyp_params, map_params):
    #calculate reward for optimal placement under hyp_reward
    hyp_obj_weights, hyp_abs_weights = hyp_params
    hyp_reward_fn = utils.RbfComplexReward(config, hyp_obj_weights, hyp_abs_weights)
    #active_utils.visualize_reward(hyp_reward_fn, "hypothesis reward")
    #get optimal placement under the hypothesis reward function and new configuration
    hyp_placement, _ = hyp_reward_fn.estimate_best_placement()

    #calculate reward for map placement under hyp_reward
    map_obj_weights, map_abs_weights = map_params
    map_reward_fn = utils.RbfComplexReward(config, map_obj_weights, map_abs_weights)
    #active_utils.visualize_reward(map_reward_fn, "map reward")
    #get optimal placement under map reward function and new configuration
    map_placement, _ = map_reward_fn.estimate_best_placement()
    #print "placement loss", np.linalg.norm(hyp_placement - map_placement)
    #plt.show()
    return np.linalg.norm(hyp_placement - map_placement)


def get_best_placement(config, map_params):
    #calculate reward for map placement under hyp_reward
    map_obj_weights, map_abs_weights = map_params
    map_reward_fn = utils.RbfComplexReward(config, map_obj_weights, map_abs_weights)
    #active_utils.visualize_reward(map_reward_fn, "map reward")
    #get optimal placement under map reward function and new configuration
    map_placement, _ = map_reward_fn.estimate_best_placement()
    return map_placement