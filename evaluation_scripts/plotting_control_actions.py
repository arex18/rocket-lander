"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: General scripts for graphing instead of notebook.
"""

from plotting.plotty import  *
import numpy as np

# action_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//pid//action_history.npy')
# res = Graphing(plot_colors=['royalblue','darkorange','limegreen'], fig_size=(9,5))
# fig = res.create_figure()
#
#
# ax = res.add_subplot(fig, 211, "", "Control Actions")
# res.add_title('Control Actions vs. Iteration Number for Test 19 and Test 23 - PID')
#
# ax2 = res.add_subplot(fig, 212, "Iteration", "Control Actions")
#
# res.plot_graph(None, action_history[18][10:400], ax)
# res.plot_graph(None, action_history[22][10:400], ax2)
# res.add_legend(['$F_e$', '$F_s$', '$\psi$'], loc=4)
#
# plt.show()

action_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation_scripts//mpc//planned_actions_1.npy')
res = Graphing(plot_colors=['royalblue','darkorange','limegreen'], fig_size=(9,5))
fig = res.create_figure()

ax = res.add_subplot(fig, 111, "", "Control Actions")
res.add_title('Control Actions vs. Iteration Number for Test 19 and Test 23 - PID')

res.plot_graph(None, action_history, ax)
res.add_legend(['$F_e$', '$F_s$', '$\psi$'], loc=4)

plt.show()