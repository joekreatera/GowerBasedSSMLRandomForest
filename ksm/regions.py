import matplotlib.pyplot as plt
import numpy as np
import pprint
import pandas as pd
from numpy.random import default_rng
import math
import random

class RegionSet():
    def __init__(self, label, attribute, do_start_value = True):
        self.label = label
        self.attribute = attribute 
        self.label_value = -1
        self.regions = [] # tuples indicating the start and end of the regions
        self.instance_bag = []
        self.last_checked_value = np.NINF
        self.possible_instances = []
        self.weights = []
        self.average_weight = 0
        self.possible_weight = 0 
        self.bags = []
        self.foreign_bags = []
        self.undecided_bag  = []
        self.interest_points = 0
        # this should be optional
        if do_start_value:
            self.add_start_value(np.NINF)
    def add_to_last_bag(self, instance_index):
        if( len(self.bags) == 0 ):
            self.bags.append([])
        self.bags[-1].append(instance_index)
    def get_min(self):
        #print(f'min regions : {self.regions}')
        return max(self.regions[0]) if self.regions[0][0] == np.NINF else min(self.regions[0])
    
    def get_max(self):
        #print(f'max regions : {self.regions}')
        
        if( self.regions[-1] == [np.NINF, np.PINF] ):
            if( len(self.regions) > 1):
                return self.regions[-2][1] # just if there is one more region that has no actual values and there is more than one region
                
        return min(self.regions[-1]) if self.regions[-1][1] == np.PINF else max(self.regions[-1]) 
        
    def get_interval_density_average(self, total_node_instances):
        s = 0
        for e in self.bags:
            # print(e)
            s += len(e)/total_node_instances
        # print("-------------------")    
        return s/(len(self.bags)+.000001)
    
        
    def get_left_right_instances_and_decision_value(self, sorted_column):
        # get the intervals where there is half the interest points (or close to it)
        interest_points_counter = 0
        all_starts = [r[0] for r in self.regions]
        all_ends = [r[-1] for r in self.regions]
        
        for r,b in zip(self.regions, self.bags):
            interest_points_counter += len(b)
            if(interest_points_counter>self.interest_points/2):
                
                last_instance = b[-1]
                print("last instance",last_instance," interest points ",self.interest_points )
                sc_idx_value = np.where(sorted_column == last_instance)
                sc_idx_value = sc_idx_value[0][0]
                # this should always happen
                return sorted_column[0:sc_idx_value+1], sorted_column[sc_idx_value+1:], r[-1], all_starts, all_ends
                
        return sorted_column,[],np.PINF, [], [] # else return the whole 
        
    def try_add(self, instance_subset, actual_instance_index_id, next_instance_index_id, list_of_interest, actual_unsupervised):
        #print(instance_subset, actual_instance_index_id, next_instance_index_id, list_of_interest, actual_unsupervised)
        actual_instance = instance_subset.loc[actual_instance_index_id]
        next_instance = instance_subset.loc[next_instance_index_id]
        actual_is_on_interest = (actual_instance_index_id in list_of_interest)
        next_is_on_interest = (next_instance_index_id in list_of_interest)
        #print(self.regions)
        #print(self.bags)
        # print(f"trying {actual_instance_index_id} {next_instance_index_id} {actual_is_on_interest} {next_is_on_interest}")
        if( len(self.regions) == 0 ): # this is the first one, start a region
            if actual_is_on_interest:
                self.regions.append([np.NINF, np.PINF])
                self.regions[-1][0] = actual_instance[self.attribute]
                self.bags.append([])
                
        if actual_is_on_interest:
            self.interest_points += 1
            self.bags[-1].append(actual_instance_index_id)
            if(not next_is_on_interest and len(self.regions) > 0 ): # the next one is different    
                self.regions[-1][1] = (next_instance[self.attribute] + actual_instance[self.attribute])/2
                self.regions.append([np.NINF, np.PINF]) # will look for the next one
                self.bags.append([]) # preprare for a following one
                self.foreign_bags.append([])
            # else:
                # print(f"Region here! {actual_instance[self.attribute]} {next_instance[self.attribute]}")
                # the worst idea 
                #self.bags[-1] += self.undecided_bag # suppose all unsupervised between two sup. points are from the same label--> this is a big jump here
                #self.undecided_bag.clear()
                #self.regions[-1][0] = (next_instance[self.attribute] + actual_instance[self.attribute])/2
        
        if not actual_is_on_interest:
            #if(actual_unsupervised): Worst idea 
            #    self.undecided_bag.append(actual_unsupervised)
            #    return # do nothing more 
            # this might be a know (supervised) or unknown labels instance (unsupervised)
            if(len(self.foreign_bags) > 0 ): # do this only after the first region of interest
                self.foreign_bags[-1].append(actual_instance_index_id)
            if( len(self.regions) == 0 ):
                self.regions.append([np.NINF, np.PINF])
                self.bags.append([])
            if next_is_on_interest: # supervised not on interest
                # self.undecided_bag.clear() , not a good idea
                self.regions[-1][0] = (next_instance[self.attribute] + actual_instance[self.attribute])/2
        
        # if next is on interest, wait for the next cycle
        # missing a closing function that ends the intervals
        
    def get_regions_instances_values_average_distance_to_interval_mean(self, instances, common_instances):
        """
        intervals AND instances on instance bag are repeating!! 
        """
        averages = []
        weights = []
        # print(f'{self.regions} {self.instance_bag}')
        for interval in self.regions:
            interval_values = []
            width = interval[1]-interval[0]
            
            for i in self.instance_bag:
                if i in common_instances:
                    val = instances.loc[i][self.attribute]
                    
                    if( interval[0] == np.NINF ):
                        if( interval[1] == np.PINF ):
                            # there is only one interval, all the instances are in this region
                            return 0,1 # min distance, max weight
                        else:
                            # just on side is defined 
                            if( val <= interval[1] ):
                                interval_values.append( 0.5  ) # no way of measuring
                    elif( interval[1] == np.PINF ):
                            if( val >= interval[0] ):
                                interval_values.append( 0.5  ) # no way of measuring
                    elif(val >= interval[0] and val <= interval[1]):
                        interval_values.append( abs(  (val-interval[0])/width - 0.5  )  )
                        
            if( len(interval_values) > 0 ):
                averages.append(  sum(interval_values)/len(interval_values) )
                weights.append(len(interval_values)/len(common_instances) ) # hopefully this is equal to common_instances
            else:
                averages.append(0.5) # max difference  
                weights.append(0)
        avg = sum(averages)/len(averages) # the smaller the better, towards 0, max=0.5 
        wts = sum(weights)/len(weights) # the bigger, the better, towards 1, min 0
        
        return avg, wts
    def calculate_proportional_weight(self, total_instances):
        self.weights = [w/(total_instances+0.0000001) for w in self.weights] # just in case total_instances are 0
        self.average_weight = sum(self.weights)/(len(self.weights)+0.0000001)
    
    def add_possible_weight(self):
        self.possible_weight+= 1 # unsupervised instances that might become part of this region
    
    def commit_possible_weight(self):
        self.weights[-1] += self.possible_weight # unsupervised instances that might become part of this region
        self.possible_weight = 0
        
    def add_weight(self):
        self.weights[-1] += 1;
        
    def get_average_weight(self):
        return self.average_weight
        
    def get_attribute(self):
        return self.attribute
        
    def get_last_checked_value(self):
        return self.last_checked_value
        
    def set_last_checked_value(self,v):
        self.last_checked_value = v
        
            
    def add_start_value(self, value, overwrite_previous = False):
        if(not overwrite_previous):
            self.regions.append( [np.NINF,np.PINF]  )
            self.weights.append(0)
        self.regions[-1][0] = value
    
    def get_last_valid_instance(self):
        last_one = self.bags[-1]
        if(len(last_one)==0):
            if(len(self.bags) > 1 ):
                last_one = self.bags[-2]
        
        if(len(last_one)>0):
            return last_one[-1] #the last of the last
        return -1 # no instance on last bag
        
    def add_end_value(self, value):
        self.regions[-1][1] = value
        
    def set_label_value(self, label_value):
        self.label_value = label_value
        
    def get_label_value(self):
        return self.label_value
    
    def set_instances(self, instance_arr):
        self.instance_bag += (instance_arr)
    def add_instance(self, instance_id):
        self.instance_bag.append(instance_id)
    def get_instances(self):
        return self.instance_bag
        
    def add_possible_instance(self, possible):
        self.possible_instances.append(possible)
    
    def commit_possible_instances(self):
        self.instance_bag + self.possible_instances
        self.clear_possible_instances()
    
    def clear_possible_instances(self):
        self.possible_instances.clear()
        
        
    def __str__(self):
        return f'label:{self.label:<10} attr:{self.attribute:<10} lv:{self.label_value:<10}  regions:{self.regions}\n' # {self.bags}


class Region():
    """
    The class will store the instance index part of the region,
    the index of the label that the region represents
    region min (start) and max (end) values
    name of the attribute that is being searched
    """
    
    def __init__(self, label, attribute, label_value = 0, start_value = np.NINF, add_instance = -1):
        self.label = label
        self.attribute = attribute 
        self.label_value = label_value
        self.start_value = start_value
        self.end_value = np.PINF
        self.instance_bag = []
        
        if(add_instance > -1):
            self.instance_bag.append(add_instance)
    
    def set_end_value(self, end_value):
        self.end_value = end_value
        
    def set_label_value(self, label_value):
        self.label_value = label_value
        
    def get_label_value(self):
        return self.label_value
        
    def add_instance(self, instance_id):
        self.instance_bag.append(instance_id)
    def get_instances(self):
        return self.instance_bag
    def __str__(self):
        return f'label:{self.label:<10} attr:{self.attribute:<10} lv:{self.label_value:<10} sv:{self.start_value:<10} ev:{self.end_value:<10} {self.instance_bag}'

  
