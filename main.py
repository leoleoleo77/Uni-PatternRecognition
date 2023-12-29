import dataMethods
from matplotlib import pyplot as plt 
import numpy as np 


# Creating dataset
a = np.random.randint(100, size =(50))

# Creating plot
fig = plt.figure(figsize =(10, 7))

plt.hist(a, bins = [0, 10, 20, 30,
					40, 50, 60, 70,
					80, 90, 100]) 


plt.title("Numpy Histogram") 
# show plot
plt.show()
