from plotting.plotty import  *
import numpy as np

# action_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation//pid//action_history.npy')
# res = Graphing(plot_colors=['royalblue','darkorange','limegreen'], fig_size=(9,5))
# fig = res.createFig()
#
#
# ax = res.addsubplot(fig, 211, "", "Control Actions")
# res.addTitle('Control Actions vs. Iteration Number for Test 19 and Test 23 - PID')
#
# ax2 = res.addsubplot(fig, 212, "Iteration", "Control Actions")
#
# res.plotGraph(None, action_history[18][10:400], ax)
# res.plotGraph(None, action_history[22][10:400], ax2)
# res.add_legend(['$F_e$', '$F_s$', '$\psi$'], loc=4)
#
# plt.show()

action_history = np.load('C://Users//REUBS_LEN//PycharmProjects//RocketLanding//control_and_ai//evaluation//mpc//planned_actions_1.npy')
res = Graphing(plot_colors=['royalblue','darkorange','limegreen'], fig_size=(9,5))
fig = res.createFig()

ax = res.addsubplot(fig, 111, "", "Control Actions")
res.addTitle('Control Actions vs. Iteration Number for Test 19 and Test 23 - PID')

res.plotGraph(None, action_history, ax)
res.add_legend(['$F_e$', '$F_s$', '$\psi$'], loc=4)

plt.show()