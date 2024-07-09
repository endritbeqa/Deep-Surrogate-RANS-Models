import numpy as np

'''
This file creates the point coordinates for the openFoam internalCloud file 
It is saved in Point_Coordinates.txt
'''

res = 32
x_range = (-0.5, 1.5)
y_range = (-1, 1)

for y in np.linspace(y_range[0], y_range[1], res, endpoint=False):
    for x in np.linspace(x_range[0], x_range[1], res, endpoint=False):
        with open('/home/blin/PycharmProjects/Thesis/Point_Coordinates.txt', 'a') as file:
            file.write('('+str(x)+' '+str(y)+' 0.5)\n')

