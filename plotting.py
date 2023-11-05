import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('SP_data_single.csv')

sns.lineplot(data=df, x='date', y='close')
plt.title('Yearly plot of stock: AAPL')
plt.xlabel('date')
plt.ylabel('close')
plt.show()