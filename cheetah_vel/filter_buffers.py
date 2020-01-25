import numpy as np
for i in range(40):
    print(i)
    buff = np.load('full_buffer_{}.npy'.format(i))
    bb = np.delete(buff, [20, 47, 68], 2)
    np.save('filtered_buffer_{}.npy'.format(i), bb)
    
