import pandas as pd

dataframe = pd.read_csv("../../module_class.csv", encoding = "cp949")
print(dataframe)
print(dataframe.info())