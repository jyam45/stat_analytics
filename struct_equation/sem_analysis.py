import pandas as pd
import semopy

class StructEquationModel:

    def __init__(self, model):
        self.text_    = model
        self.model_   = semopy.Model(model)
        self.result_  = None
        self.inspect_ = None
        self.stats_   = None
        self.graph_   = None

    def fit(self, data):
        self.result_  = self.model_.fit(data)
        self.inspect_ = self.model_.inspect(std_est=True)
        self.stats_   = semopy.calc_stats(self.model_)
        return self.result_
    
    def model(self):
        return self.model_
    
    @staticmethod
    def __change_viewsize(nx,ny):
        pd.set_option('display.max_columns',ny)
        pd.set_option('display.max_rows',nx)

    def plot(self,filename,engine="dot"):
        self.graph_  = semopy.semplot(self.model_,filename, #inspection=self.inspect_, 
                                      plot_covs=True,std_ests=True,engine=engine)
        return self.graph_
    
    def __str__(self):
        x = ""
        if self.text_ is not None:
            x += "Equations:\n"
            x += str(self.text_)
            x += "\n\n"
        if self.result_ is not None:
            x += "Results:\n"
            x += str(self.result_)
            x += "\n\n"
        if self.inspect_ is not None:
            x += "Inspect:\n"
            self.__change_viewsize(len(self.inspect_.index),len(self.inspect_.columns))
            x += str(self.inspect_)
            x += "\n\n"
        if self.stats_ is not None:
            x += "Fit Index:\n"
            self.__change_viewsize(len(self.stats_.columns),len(self.stats_.index))
            x += str(self.stats_.T)
            x += "\n\n"
        if self.graph_ is not None:
            x += "DOT source:\n"
            x += str(self.graph_)
            x += "\n\n"
        return x

