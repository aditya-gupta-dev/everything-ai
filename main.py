import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

db = pd.read_csv('datasets/PLTR_dataset.csv')

date = np.array(db['Date'])
price = np.array(db['Price'])

plt.plot(date, price)
plt.savefig('img.png')
