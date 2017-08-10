import numpy as np

def get_raw_data(input_name, writebox, spaces=False):

    f = open('tasks/write_data/'+input_name+'.txt', 'r')
    row = f.readline()

    points = []
    for row in f:
        points.append(row.strip('\n').split(','))
    f.close()

    points = np.array(points, dtype='float')
    points = points[:, :2]

    # need to rotate the points
    theta = np.pi/2.
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    for ii in range(points.shape[0]):
        points[ii] = np.dot(R, points[ii])

    # need to mirror the x values
    for ii in range(points.shape[0]):
        points[ii, 0] *= -1

    # center numbers 
    points[:, 0] -= np.min(points[:,0])
    points[:, 1] -= np.min(points[:,1])

    # normalize 
    # TODO: solve weird scaling for 1 and l, and 9
    points[:, 0] /= max(points[:,0])
    points[:, 1] /= max(points[:,1])

    # center numbers 
    points[:, 0] -= .5 - (max(points[:,0]) - min(points[:, 0])) / 2.0

    points[:,0] *= 5.0 / 6.0 * (writebox[1] - writebox[0]) 
    points[:,1] *= (writebox[3] - writebox[2])

    if input_name in ('1'):
        points[:, 0] /= 15.
    if input_name in ('s'):
        points[:, 0] /= 5.
    if input_name in ('9'):
        points[:, 0] /= 2.
    if input_name in ('e','o','w','r'):
        points[:, 1] /= 2.

    points[:,0] += writebox[0]
    points[:,1] += writebox[2]

    return points 

def get_single(**kwargs):
    """Wrap the number with np.nans on either end
    """

    num = get_raw_data(**kwargs)
    new_array = np.zeros((num.shape[0]+2, num.shape[1]))
    new_array[0] = [np.nan, np.nan]
    new_array[-1] = [np.nan, np.nan]
    new_array[1:-1] = num

    return new_array

def get_sequence(sequence, writebox, spaces=False):
    """Returns a sequence 

    sequence list: the sequence of integers
    writebox list: [min x, max x, min y, max y]
    """

    nans = np.array([np.nan, np.nan])
    nums= nans.copy()

    if spaces is False:
        each_num_width = (writebox[1] - writebox[0]) / float(len(sequence))
    else: 
        each_num_width = (writebox[1] - writebox[0]) / float(len(sequence)*2 - 1)

    for ii, nn in enumerate(sequence):

        if spaces is False:
            num_writebox = [writebox[0] + each_num_width * ii , 
                            writebox[0] + each_num_width * (ii+1), 
                            writebox[2], writebox[3]]
        else:
            num_writebox = [writebox[0] + each_num_width * 2 * ii , 
                            writebox[0] + each_num_width * 2 * (ii+.5), 
                            writebox[2], writebox[3]]
        if isinstance(nn, int):
            nn = str(nn)
        num = get_raw_data(nn, num_writebox)
        nums = np.vstack([nums, num, nans])

    return nums 


### Testing code ###
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    files=['h','e','l','l','o','w','o','r','l','d']
    nums = get_sequence(files, writebox=[-1,1,0,1], spaces=False)
    plt.plot(nums[:,0], nums[:,1])
    plt.show()

