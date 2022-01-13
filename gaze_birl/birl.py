# -*- coding: utf-8 -*-
#uses absolute and relative RBFs to create reward function

import utils
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import warnings

#normalizes the reward weights so l1 norm is 1
#uses softmax sampling to approximate the partition function

class BIRL():
    def __init__(self, num_obj_weights, num_abs_weights, beta=10.0, num_steps=100, step_std=0.2, burn=50, skip=10):
        self.demonstrations = []
        self.beta = beta
        self.num_steps = num_steps
        self.step_std = step_std
        self.burn = burn
        self.skip = skip
        self.num_obj_weights = num_obj_weights
        self.num_abs_weights = num_abs_weights
        self.mcmc_chain = []
        self.demonstrations = []
        self.map_obj_weights = None
        self.map_abs_weights = None
        self.init_obj_weights = np.zeros(self.num_obj_weights)
        self.init_abs_weights = np.zeros(self.num_abs_weights)
        self.run_before = False #if mcmc has been run before we should seed the MCMC chain with the map from last time
        
    ''' a demo is a list of object centroids and an (x,y) placement'''
    def add_demonstration(self, centroids, placement):
        self.demonstrations.append((centroids,placement))
        
    ''' a demo is a list of object centroids, an (x,y) placement and gaze fixation time per object'''
    def add_gaze_demonstration(self, centroids, placement, look_times):
        self.demonstrations.append((centroids,placement,look_times))
        
    def run_inference(self, gaze=False):
        #either randomly seed or used MAP from previous iteration
        if not self.run_before:
            curr_obj_weights, curr_abs_weights = self.generate_proposal_weights(self.init_obj_weights, self.init_abs_weights)
        else: 
            curr_obj_weights, curr_abs_weights = self.get_map_params()
    
        self.mcmc_chain = []
        print "running birl"
        #check if there are demos
        assert(len(self.demonstrations) > 0)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
                  
        #compute log likelihood over demonstrations
        if (not gaze):
            curr_ll =  self.log_likelihood(curr_obj_weights, curr_abs_weights) # standard BIRL
        else:
            curr_ll =  self.gaze_log_likelihood(curr_obj_weights, curr_abs_weights) # gaze-augmented BIRL
        best_ll = -np.inf
        
        #run MCMC
        accept_cnt = 0
        for step in range(self.num_steps):
            #compute proposal and log likelihood
            prop_obj_weights, prop_abs_weights = self.generate_proposal_weights(curr_obj_weights, curr_abs_weights)
            if (not gaze):
                prop_ll = self.log_likelihood(prop_obj_weights, prop_abs_weights)
            else:
                prop_ll = self.gaze_log_likelihood(prop_obj_weights, prop_abs_weights)

            prob_accept = min(1.0, np.exp(prop_ll - curr_ll))
            if np.random.rand() < prob_accept:
                accept_cnt += 1
                #accept and add to chain
                self.mcmc_chain.append((prop_obj_weights.copy(), prop_abs_weights.copy()))
                curr_ll = prop_ll
                curr_obj_weights = prop_obj_weights
                curr_abs_weights = prop_abs_weights
                if prop_ll > best_ll:
                    best_ll = prop_ll
                    self.map_obj_weights = prop_obj_weights.copy()
                    self.map_abs_weights = prop_abs_weights.copy()
            else:
                self.mcmc_chain.append((curr_obj_weights.copy(), curr_abs_weights.copy()))
                
        #update init weights to map of last run
        self.init_obj_weights, self.init_abs_weights = self.get_map_params()
        
    def get_mcmc_chain(self):
        return self.mcmc_chain
        
    def get_map_params(self):
        return self.map_obj_weights, self.map_abs_weights
        
    def get_mean_params(self):
        mean_obj = np.zeros(self.num_obj_weights)
        mean_abs = np.zeros(self.num_abs_weights)
        for i in range(self.burn, len(self.mcmc_chain), self.skip):
            r_obj, r_abs = self.mcmc_chain[i]
            mean_obj += r_obj
            mean_abs += r_abs

        #normalize
        weight_l1_norm = 0
        for w1 in mean_obj:
            weight_l1_norm += np.abs(w1)
        for w2 in mean_abs:
            weight_l1_norm += np.abs(w2)
        for i in range(len(mean_obj)):
            mean_obj[i] /= weight_l1_norm
        for i in range(len(mean_abs)):
            mean_abs[i] /= weight_l1_norm            
        
        return mean_obj, mean_abs
        
    
    def log_likelihood(self, hyp_obj_weights, hyp_abs_weights, granularity = 10):
        log_sum = 0.0
        for centers, placement, _ in self.demonstrations:
            #to compute the optimal placement under the demo config
            #make new rbf with hypothesis weights and widths
            hyp_rbf = utils.RbfComplexReward(centers, hyp_obj_weights, hyp_abs_weights)
            placement_reward = hyp_rbf.get_reward(placement)
                      
            xspace = np.linspace(0,1,granularity)
            yspace = np.linspace(0,1,granularity)
            pairs = [(xi,yi) for xi in xspace for yi in yspace]
            
            # pairs
            Z_exponents = []
            for pair in pairs:
                Z_exponents.append(self.beta * hyp_rbf.get_reward(pair))
            #Z_exponents
            log_sum += self.beta * placement_reward - scipy.misc.logsumexp(Z_exponents)
        return log_sum



    def gaze_log_likelihood(self, hyp_obj_weights, hyp_abs_weights, granularity = 10):
        log_sum = 0.0
        for centers, placement, gaze in self.demonstrations:
            #to compute the optimal placement under the demo config
            #make new rbf with hypothesis weights and widths
            hyp_rbf = utils.RbfComplexReward(centers, hyp_obj_weights, hyp_abs_weights)
            placement_reward = hyp_rbf.get_reward(placement)
                      
            xspace = np.linspace(0,1,granularity)
            yspace = np.linspace(0,1,granularity)
            pairs = [(xi,yi) for xi in xspace for yi in yspace]

            Z_exponents = []
            for pair in pairs:
                Z_exponents.append(self.beta * hyp_rbf.get_reward(pair))

            num_objects = len(hyp_obj_weights)/5
            obj_weights = [0] * num_objects
            look_times = [0] * num_objects
            for i in range(num_objects):
                if sum(hyp_obj_weights)!= 0:
                    obj_weights[i] = sum(hyp_obj_weights[i*5:(i+1)*5])/(sum(hyp_obj_weights))
                    
                else:
                    obj_weights[i] = 1000000

                look_times[i] = gaze[i]/sum(gaze)

            gaze_term = 0
            for i in range(num_objects):
                for j in range(num_objects):
                    if i==j:
                        continue
                    else:
                        indicator_term = (obj_weights[j]>obj_weights[i] and look_times[j]<look_times[i])
                                         
                        if(look_times[j]!=0):
                            ratio = look_times[i]/look_times[j]
                        else:
                            ratio = 1000000
                        gaze_term += indicator_term * ratio
                    

            log_sum += self.beta * placement_reward - scipy.misc.logsumexp(Z_exponents) - gaze_term
        return log_sum
        
    #generate normalized weights sum L1 =1
    def generate_proposal_weights(self, weights1, weights2):
        new_weights1 = weights1.copy()
        new_weights1 += np.random.normal(0, self.step_std, weights1.shape)
        new_weights2 = weights2.copy()
        new_weights2 += np.random.normal(0, self.step_std, weights2.shape)
        
        #normalize
        weight_l1_norm = 0
        for w1 in new_weights1:
            weight_l1_norm += np.abs(w1)
        for w2 in new_weights2:
            weight_l1_norm += np.abs(w2)
        for i in range(len(new_weights1)):
            new_weights1[i] /= weight_l1_norm
        for i in range(len(new_weights2)):
            new_weights2[i] /= weight_l1_norm
            
        return new_weights1, new_weights2

        
def run_two_obj_birl_test():
    num_objs = 2
    weights = 2.0*np.random.rand(num_objs)-1.0
    widths = np.random.rand(num_objs)
    centers = np.random.rand(num_objs,2)
    #print "true weights", weights
    #print "true widths", widths
    
    rbf = active_utils.RbfReward(centers, weights, widths)

    rbf.plot_heat_map()
    
    best_x, reward = rbf.estimate_best_placement(plot=False)
    plt.plot(best_x[0], best_x[1],'*',markersize=30)
    #print "best x", best_x, "best reward", reward
    
    #plot centers of objects
    for c in centers:
        plt.plot(c[0],c[1],'o',markersize=20)
    plt.title("demonstration")


    birl = BIRL(num_objs)
    birl.add_demonstration(centers, best_x)
    birl.run_inference()
    map_weights, map_widths = birl.get_map_params()
    print "map weights", map_weights
    print "map widths", map_widths
    rbf_map = active_utils.RbfReward(centers, map_weights, map_widths)

    rbf_map.plot_heat_map()
    
    map_x, reward = rbf_map.estimate_best_placement(plot=False)
    plt.plot(map_x[0], map_x[1],'*',markersize=30)
    #plot centers of objects
    for c in centers:
        plt.plot(c[0],c[1],'o',markersize=20)
    plt.title("map estimate")
    #print "best x", map_x, "best reward", reward
    plt.show()
        
if __name__=="__main__":
    #run_two_obj_birl_test()
    birl = BIRL(3, 4, beta=10.0, num_steps=100, step_std=0.1)
    w1 = np.zeros(3)
    w2 = np.zeros(4)
    for i in range(3):
        print "---------"
        w1, w2 = birl.generate_proposal_weights(w1, w2)
        print w1
        print w2
        print np.linalg.norm(np.append(w1,w2),1)
        
