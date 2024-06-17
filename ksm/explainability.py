import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
import numpy

def string_to_list(r):
            x = r["pred_label_probs"]
            v = [float(i) for i in x]
            return v

def convert_dict(x):
    #print(x[0])
    d = x[0]
    #print(d)
    return d

def generate_explainability_files(rules_df, labels_distrib,  labels_distrib_supervised, trees_qty) :

    
    plp = rules_df[ rules_df["is_leaf"] == True]
    np_plp = numpy.array(plp["pred_label_probs"].values)
    np_avg = np_plp.mean(axis=0)

    labels_distribution_df = pd.DataFrame( [ labels_distrib.values , labels_distrib_supervised.values , np_avg ] , index=["all_labels","supervised_labels","model_labels"] )
    print(labels_distribution_df) 

    rules_df.to_csv("rules.csv")

    leafs_df = rules_df[ rules_df["is_leaf"] == True ] 
    leafs_df.reset_index(inplace=True)
    # print(leafs_df)
    leafs_df = leafs_df[ ["id","tree_id","pred_label_probs"] ]
    leafs_df.reset_index(inplace=True , drop=True)

    sep_df = leafs_df.apply( string_to_list, result_type="expand" , axis=1)
    sep_df.rename( columns =  {i:f'label_{i}' for i in sep_df.columns }, inplace=True )

    df_pp = pd.concat(  [ leafs_df  , sep_df ] ,axis = 1 )
    df_pp.drop(columns=["pred_label_probs"] ,  inplace=True )

    grouped = df_pp[ [f"{i}" for i in sep_df.columns] + ["id", "tree_id"] ].groupby("id").mean()
    trees = grouped["tree_id"]/max(grouped["tree_id"])
    np_hsv_colors = trees.values
    np_sat_color = numpy.ones(shape=np_hsv_colors.shape)
    np_val_color = numpy.ones(shape=np_hsv_colors.shape)*0.9

    hsv = numpy.array( [np_hsv_colors, np_sat_color, np_val_color] ).T
    rgb = hsv_to_rgb(hsv)
    #print(rgb)
    opacity = numpy.ones(shape = [rgb.shape[0] , 1] )*0.1
    rgb = numpy.hstack( [rgb,opacity] )

    grouped.drop(columns=["tree_id"] , inplace=True)


    plt.figure(figsize=(15,8))
    parallel_plot = sns.lineplot(data=grouped.transpose(),
                                markers=False ,
                                dashes=False,
                                palette  = list(rgb) , legend=False)
    
    plt.title('Label Distribution Plot')
    plt.yticks([])
    plt.savefig("label_distribution_plot.png")
    plt.close()



    limits = rules_df[ rules_df["is_leaf"] == True][ ["mm_limits"]]
    A = limits.apply(convert_dict, result_type="expand", axis=1)
    cols_dict = dict()
    col_index = 0

    fig = plt.figure( figsize=(20,10) )
    
    for index,col in A.items():
        cols_dict[col.name] = {'index':col_index, 'data' : [] }
        col_index += 1

        for elem in col.items():
            if not (elem[1] is numpy.nan):
                cols_dict[col.name]["data"].append(elem[1])

        # upon finishing, normalize
        numpy_data = numpy.array(cols_dict[col.name]["data"])
        min_c = numpy_data.min()
        max_c = numpy_data.max()
        diff = max_c-min_c
        # print(numpy_data)
        numpy_data = (numpy_data - min_c)/diff
        #print(f"m mx  {min} {max} {diff} " )

        for d in numpy_data:
            plt.bar(col_index-1, height=d[1]-d[0] , width=0.8, bottom=d[0] , color=(0.7,0.7,0.7, abs(200-trees_qty)/2000.0)  ) # , label = f"at {col_index}"
            # print(d)
        
    axes = fig.axes

    axes[0].set_xticks( [i for i in range(0,len(cols_dict))], labels=[f"col_{i}" for i in range(0,len(cols_dict))] , rotation='vertical')

    plt.savefig("attribute_distribution_plot.png")
    plt.close()