class VarianceReductionChampionProposal:
    def __init__(self,  point_index,
                        left_indexes,
                        right_indexes,
                        point_value,
                        variance_reduction_coefficient,
                        col_name):
        self.best_split_point_index = point_index
        self.left = left_indexes
        self.right = right_indexes
        self.value = point_value
        self.variance_coefficient = variance_reduction_coefficient
        self.column_name = col_name
        self.compatibility_average= -1
        self.compatibility_average_left= -1
        self.compatibility_average_right = -1

    def set_compatibility_average(self, p):
        self.compatibility_average= p

    def set_compatibility_average_left(self, p):
        self.compatibility_average_left = p

    def set_compatibility_average_right(self, p):
        self.compatibility_average_right = p

    def __str__(self):
        return f'{self.best_split_point_index}___,___{self.left}___,___{self.right} @ {self.value} {self.variance_coefficient} on {self.column_name}'
