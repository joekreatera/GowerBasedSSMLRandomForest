import numpy
import pandas
class UD3_5Clustering:
    """
        Really basic class that divides a dataset in two clusters according to:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9035478
        
        NEEDS PANDAS!
    """
    def __init__(self):
        self.column = None
        self.value = None
        self.quality = 0
    
    def get_best_split(self, col, dataset):
        # obtaine the column from dataset, ordered by column
        # print column_name
        #print(col.name)
        
        dataset_column = dataset[col.name]
        
        ordered_data = dataset_column.iloc[col.to_numpy()]
        #print(ordered_data)
        left_sum = ordered_data.iloc[0]+ordered_data.iloc[1]
        right_sum = ordered_data.sum() - left_sum
        total_elements = ordered_data.shape[0]
        
        best_q1 = 0
        best_k = -1
        best_left_mean  = 0
        best_right_mean = 0
        best_value = numpy.NINF
        
        for val,i in zip(ordered_data[1:-1], list(range(1,total_elements-2)) ):
            # is is next index
            k=i+1
            left_mean = left_sum/k
            right_mean =  right_sum/(total_elements-k)
            l1 = ( abs(ordered_data.iloc[0] - right_mean) - abs(ordered_data.iloc[0] - left_mean) )/ max(  abs(ordered_data.iloc[0] - right_mean) , abs(ordered_data.iloc[0] - left_mean)  ) 
            l2 = ( abs(val - right_mean) - abs(val - left_mean) )/ max(  abs(val - right_mean) , abs(val - left_mean)  ) 
            r1 = ( abs(ordered_data.iloc[k] - left_mean) - abs(ordered_data.iloc[k] - right_mean) )/ max(  abs(ordered_data.iloc[k] - right_mean) , abs(ordered_data.iloc[k] - left_mean)  ) 
            r2 = ( abs(ordered_data.iloc[total_elements-1] - left_mean) - abs(ordered_data.iloc[total_elements-1] - right_mean) )/ max(  abs(ordered_data.iloc[total_elements-1] - right_mean) , abs(ordered_data.iloc[total_elements-1] - left_mean)  ) 
            q1 = (k*(l1+l2)*0.5 + (total_elements-k)*(r1+r2)*0.5)/total_elements
            #q2_left = ( abs(val - right_mean) - abs(val - left_mean) )/ max(  abs(val - right_mean) , abs(val - left_mean)  ) 
            # print(q1)
            if( q1 > best_q1):
                best_q1 = q1
                best_k = k
                best_left_mean = left_mean
                best_right_mean = right_mean
                best_value = val
            # print(val , "___" ,left_sum, "/" , left_mean, " ___ ", right_sum , "/" , right_mean , "=>" , q1)
            left_sum += val
            right_sum -= val
        # sum all the quality 2 measures from 0  to k-1 
        #print(f"{best_left_mean} {best_right_mean} {best_value} ")
        left_left_col = ((ordered_data[ordered_data<=best_value] - best_left_mean).abs()).to_numpy()
        left_right_col = ((ordered_data[ordered_data<=best_value] - best_right_mean).abs()).to_numpy()        
        left_best = (( left_right_col - left_left_col  )/ numpy.maximum(left_left_col,left_right_col)).sum()
        #print(left_best)
         
        # sum all the quality 2 measures from k to total_elements-1
        right_left_col = ((ordered_data[ordered_data>best_value] - best_left_mean).abs()).to_numpy()
        right_right_col = ((ordered_data[ordered_data>best_value] - best_right_mean).abs()).to_numpy()        
        right_best = (( right_left_col - right_right_col  )/ numpy.maximum(right_left_col,right_right_col)).sum()
        #print(right_best)
        
        # divide the sum of previous by total elements
        total_best=(left_best+right_best)/total_elements
        #print(total_best)
        #print("----------------")
        return {'value':best_value,'quality_eval':total_best} # return the evaluation of quality and the value to divide
    
    def fit(self,dataset):
        """
        prepares the boundary decision for the rows to be predicted -> 0 or 1
        the dataset should come only with the columns to be used 
        """
        if( not  (type(dataset) is  pandas.DataFrame) ):
            dataset = pandas.DataFrame(data = dataset)
        arr = dataset.to_numpy()
        order = arr.argsort(axis=0)
        #print(order)
        #print(dataset)
        order_df = pandas.DataFrame(data=order, columns=dataset.columns) # the index is unimportant here as the data is the actual order 
        res = order_df.apply(self.get_best_split, axis=0 ,result_type='expand', dataset=dataset ).T
        res = res.sort_values(by='quality_eval')
        #print(res.tail(1))
        self.column = res.index[0]
        self.value = res['value'].iloc[0]
        self.quality = res['quality_eval'].iloc[0]
        
    def predict(self, rows, tree_params_return = False):
        """
        predict the cluster of the rows according to the best column 
        """
        #print("-------------------")
        #print(rows)
        result = []
        if self.column is None:
            raise Exception("Call fit() first")
        else:
            for index, row in rows.iterrows():
                #print(row)
                val = row[self.column]
                #print(val) 
                if( val <= self.value):
                    result.append(0)
                else:
                    result.append(1)
        return numpy.array(result)
    
    def fit_predict(self, X):
        """
        fits and return the cluster label of X
        """
        self.fit(X)
        if(not type(X) is pandas.DataFrame ):
            X = pandas.DataFrame(data=X)
        
        return self.predict(X)

    def get_tree_params(self):
        return self.column, self.value, self.quality
