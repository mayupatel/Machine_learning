#This code is all about building a data model through a class function.

#Importing the libraries from python to use it here
import numpy as np

#writing a class in python for data modelling
class data_model:
    """
    Class for creating data models.

    Attributes:
    """
    
    def __init__(self, data):
        self.data = np.array([]) #taking the dataset as a numpy array
        self.feature_names = []  # taking features 
        self.target = np.array([]) # taking target
        self.target_names = []   # taking target names

        self.make_data_model(data)
    
    #building a data model
    def make_data_model(self, data_in):
        num_of_variables = data_in.shape[1] - 1 # using shape to select the variable numbers
        target = data_in.iloc[:, num_of_variables] 
        data = data_in.iloc[:, 0:num_of_variables]

        target_names = np.unique(target) # taking unique target names
        feature_names = (data_in.keys())[0:num_of_variables] # features names
        
        # all the data values, features and target are stored into variable for using it in other analysis.
        self.data = data.values
        self.feature_names = feature_names.tolist()
        self.target = target.values
        self.target_names = target_names.tolist()


