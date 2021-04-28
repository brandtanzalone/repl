import pandas as pd
from typing import List

class Model():

    def __init__(self):
        self.pred_table = pd.DataFrame()

    def train(self, data: pd.DataFrame):
        #group the data by month
        self.pred_table = data.groupby(['month']).mean().reset_index()
        return self

    def predict(self, month: int):
        return [self.pred_table.loc[month, 'standardized_sales'], self.pred_table.loc[month, 'standardized_traffic']]

        