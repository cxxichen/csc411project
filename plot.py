import numpy as np
import matplotlib.pyplot as plt

depth = [4, 8, 12, 16, 20, 24]
f1 = [0.9172, 0.9485, 0.9480, 0.9446, 0.9416, 0.9431]

plt.plot(depth, f1, 'r-', label='Validation Set')
plt.xlabel('Maximum Depth of Decision Tree')
plt.ylabel('F1-Measure')
plt.legend(loc='lower right')
plt.xlim([0,25])
plt.ylim([0.9, 0.95])
plt.title('F1-Measure Vs different value of max depth')
plt.show()
