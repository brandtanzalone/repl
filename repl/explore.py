import pandas as pd
from datetime import datetime
from sklearn import model_selection
import matplotlib.pyplot as plt

def either_missing(row):
    if row['Value_x'] == False and row['Value_y'] == False:
        return False
    else:
        return True


if __name__ == "__main__":
    #load the data
    sales = pd.read_csv('REPL_ML_Exercise/training_Sales.csv')
    traffic = pd.read_csv('REPL_ML_Exercise/training_Traffic.csv')
    
    print(len(sales))
    print(len(traffic))
    
    #looks like data, lets aggregate the tables joining on time
    data_inner = sales.merge(traffic, on='Date')
    print(len(data_inner))

    #inner join tells us we have missing value, try an outer
    data_outer = sales.merge(traffic, on='Date', how='outer')

    #lets look at just the missing values
    missing = data_outer.isna()
    missing['missing'] = missing.apply(either_missing, axis=1)
    just_missing = data_outer[missing['missing'] == True]
    
    #missing a large number of observations
    #no observations of traffic before 2015
    #sporadically missing sales information

    #simply exclude ~17k observations
    #impute observations after 2015

    #look at some quick stats
    dmax = data_outer.max()
    dmin = data_outer.min()
    dmean = data_outer.mean()
    
    #replace all the na values with appropriate means
    filled_data = data_outer.fillna(dmean)
    
    #convert Date to datetime
    filled_data['datetime'] = filled_data.apply(lambda x: datetime.strptime(x['Date'], '%Y-%m-%d %H:%M:%S'), axis=1)
    
    #trim the leading entries before 2015
    trim_to = datetime(year=2015, month=1, day=1)
    trim_after = datetime(year=2018, month=4, day=30)
    trimmed_filled_data = filled_data[filled_data['datetime'] > trim_to]
    trimmed_filled_data = trimmed_filled_data[trimmed_filled_data['datetime'] < trim_after]
    
    #make table more explicit
    final = trimmed_filled_data.rename(columns={'Value_x': 'Sales', 'Value_y': 'Traffic'})
    final['year'] = final.apply(lambda x: x['datetime'].year, axis = 1)
    final['month'] = final.apply(lambda x: x['datetime'].month, axis = 1)

    #we want to predict sales in the next month
    #therefore we need our data to have notion of the next month
    agg_final = final.groupby(['year', 'month']).sum().reset_index()

    #plot sales and traffic to investigate
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(agg_final['month'], agg_final['Sales'])
    ax.set_xlabel('month')
    ax.set_ylabel('sales')
    plt.show()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.scatter(agg_final['month'], agg_final['Traffic'])
    ax.set_xlabel('month')
    ax.set_ylabel('traffic')
    plt.show()

    #standardize the data
    sales_mean = agg_final['Sales'].mean()
    sales_std = agg_final['Sales'].std()

    traffic_mean = agg_final['Traffic'].mean()
    traffic_std = agg_final['Traffic'].std()

    agg_final['standardized_sales'] = agg_final.apply(lambda x: (x['Sales'] - sales_mean) / sales_std, axis=1)
    agg_final['standardized_traffic'] = agg_final.apply(lambda x: (x['Traffic'] - traffic_mean) / traffic_std, axis=1)
    

    #write the aggregated data to disk
    agg_final.to_csv('data/data.csv')

    #train/test split
    train, test = model_selection.train_test_split(agg_final, test_size= .1)
    train.to_csv('data/train.csv')
    test.to_csv('data/test.csv')