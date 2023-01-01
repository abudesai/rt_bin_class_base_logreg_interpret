
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore') 

from sklearn.linear_model import LogisticRegression as LogRegSklearn
from interpret.glassbox import LogisticRegression


model_fname = "model.save"
global_explanations_fname = "global_explanations.csv"
global_explanations_chart_fname = "global_explanations.png"

MODEL_NAME = "bin_class_base_log_reg_interpret"

COST_THRESHOLD = float('inf')


class Classifier(): 
    
    def __init__(self, feature_names, penalty = "elasticnet", C = 1.0, l1_ratio=0.5, **kwargs) -> None:
        self.feature_names = feature_names
        self.penalty = penalty  
        self.C = C  
        self.l1_ratio = np.float(l1_ratio)    
        self.model = self.build_model()     
        
        
    def build_model(self): 
        #This is the Logisitc Regression from the interpret package, not from Sklearn
        model = LogisticRegression(
            feature_names = self.feature_names, 
            linear_class = LogRegSklearn,          # this Logistic Regression is from Sklearn
            penalty= self.penalty, 
            C= self.C, 
            l1_ratio= self.l1_ratio, 
            solver='saga',          # have to do this because lbfgs supports only 'l2' or 'none' penalties, we are using elasticnet which has l1
            random_state=0)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(
            X= train_X, 
            y= train_y)            
        
    def explain_global(self, name): 
        return self.model.explain_global(name=name)

    def predict_proba(self, X, verbose=False): 
        preds = self.model.predict_proba(X)
        return preds 

    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)       

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))
        self._save_global_explanations(model_path=model_path)
        


    def _save_global_explanations(self, model_path):
        global_explanations = self.model.explain_global(name="logreg")
        data = global_explanations.data()
        df = pd.DataFrame()
        # create list of feature names - "extra" contains the intercept
        # read documentation of interpret ml global_explanations for details        
        df['feature'] = data['names'] + [ data['extra']['names'][0] ]
        df['score'] = data['scores'] + [ data['extra']['scores'][0] ]      
        df.sort_values(by=['score'], inplace=True, ascending=False)
        df.to_csv(os.path.join(model_path, global_explanations_fname), index=False, float_format='%.4f')
        save_plot_of_explanations(df['score'], df['feature'], model_path)
        


    @classmethod
    def load(cls, model_path): 
        model = joblib.load(os.path.join(model_path, model_fname))
        return model


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    model = Classifier.load(model_path)  
    return model



def save_plot_of_explanations(vals, labels, model_path):    
    height = 2. + len(vals)*0.3
    colors = ['rosybrown' if x < 0 else 'steelblue' for x in vals]
    plt.figure(figsize=(16, height), dpi=80) 
    plt.barh(labels,vals, color=colors)  
    plt.xlabel('score', fontsize=18); plt.ylabel('feature', fontsize=16)
    plt.yticks(labels, fontsize=14); plt.xticks(fontsize=14)
    plt.title('Global Feature Impact', fontdict={ 'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(model_path, global_explanations_chart_fname))
