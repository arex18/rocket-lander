
import numpy as np
from matplotlib import pyplot as plt

a = np.load(r'C:\Users\suntz\rocket_lander\rocket_proj\rocket-lander\DDPG_update\reward_history\reward_history_1488ep.npy')
b = np.load(r'C:\Users\suntz\rocket_lander\rocket_proj\rocket-lander\DDPG_update\reward_history\reward_history.npy')

c = np.concatenate((a[:1400],b[:600])) # didn't continue on as expected

avg = []
n=100
for idx in range(len(a)):
    end = idx+1
    start = np.clip(end-n, 0, None)
    print(f'start {start}, end: {end}')
    avg.append(np.mean(a[start:end]))

#print(len(avg))
x=np.arange(len(avg))
plot = plt.plot(x,avg)
plt.title('Average Rewards for DDPG training (n=100)')
plt.xlabel('Episode number')
plt.ylabel('Episode Reward')
plt.grid(True, which='both')
plt.show()
