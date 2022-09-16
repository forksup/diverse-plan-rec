import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar
CAP = [0.6911592277026499, 0.6910810599546627,0.6893222856249511]
DBN = [0.6811537559602908]
CAP_COUNT = [0.6354601470880541, 0.627708209103558, 0.6275094414629299, 0.6268468826608361]
MARKOV_COUNT = [0.6354601470880541, 0.627708209103558, 0.6275094414629299, 0.6268468826608361]



# Set position of bar on X axis
br1 = np.arange(len(CAP))
br2 = [br1[0] + barWidth]

br3 = [x + barWidth for x in br1]
br3 = [br1[0]+barWidth] + br3
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, CAP, width=barWidth,
        edgecolor='grey', label='Cap Model')
plt.bar(br2, DBN,  width=barWidth,
        edgecolor='grey', label='DBN')
plt.bar(br3, CAP_COUNT,  width=barWidth,
        edgecolor='grey', label='CAP_COUNT')
plt.bar(br4, MARKOV_COUNT, width=barWidth,
        edgecolor='grey', label='MARKOV_COUNT')

# Adding Xticks
plt.xlabel('Network count / Order', fontweight='bold', fontsize=15)
plt.ylabel('Students passed', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(CAP))],
            range(len(CAP)))

plt.legend()
plt.show()