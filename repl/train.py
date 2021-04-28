import model
import pandas as pd
import pickle


if __name__ == "__main__":
    #load data from csv
    train_data = pd.read_csv('data/train.csv')

    #instantiate model
    model = model.Model()

    #prepare data from training
    features = train_data[['year', 'month']].values
    targets = train_data[['standardized_sales', 'standardized_traffic']].values

    #train model
    trained_linear_model = model.train(train_data)
    #save model to disk
    pickle.dump(trained_linear_model, open('models/mean_avg.sav', 'wb'))