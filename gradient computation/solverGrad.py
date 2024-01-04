import scipy as sp
import numpy as np
import mesh as mesh
import argparse
import matplotlib.pyplot as plt
from src import base as base
from src import grad as grad
import os

# Define initial and boundary conditions
def initialCond(time, x):
	Nfields = 1
	qi = np.zeros(Nfields, float)
	qi = x[0]**2 + 3*x[1]
	return qi


def boundaryCond(bc, time, x, qM):
	Nfields = 1
	qi = np.zeros((Nfields,1), float)
	if(bc==1 or bc ==2):
		qi = x[0]**2 + 3*x[1]
	elif(bc==3):
		qi = qM
	return qi

#-----PROBLEM SETUP:
# Create parser 
parser = argparse.ArgumentParser(prog='Gradient',description='Compute Gradient using FVM',
                    epilog='--------------------------------')

# Set default values for gradient computation
parser.add_argument('--meshFile', default ='data/mixed.msh', help='mesh file to be read')

parser.add_argument('--method', type=str, default='GREEN-GAUSS-CELL', \
	choices=['GREEN-GAUSS-CELL', 'GREEN-GAUSS-NODE', 'LEAST-SQUARES', 'WEIGHTED-LEAST-SQUARES'],\
	help='Gradient reconstruction algorithm')

parser.add_argument('--Correct', type=bool, default=False, help='Correction for GGCB reconstruction')
parser.add_argument('--Nfields', type=int, default=1, help='Number of fields')
parser.add_argument('--dim', type=int, default=2, choices= [2,3], help='Dimension of the proble ')
parser.add_argument('--IC', type = initialCond, default = initialCond, help='Initial condition function')
parser.add_argument('--BC', type = boundaryCond, default = boundaryCond, help='Boundary condition function')
args = parser.parse_args()
# Create Boundary Field Defined on Boundary Faces 
time = 0.0; 

#-----CREATE MESH and CONNECTIVITY----------#
# Create mesh class and read mesh file
# Read mesh file and setup geometry and connections
msh = base(args.meshFile)

# Create an element field and assign initial condition
Qe = msh.createEfield(args.Nfields)
exactgradQ = np.zeros((msh.Nelements,1, msh.dim), float)
for elm,info in msh.Element.items():
    x = info['ecenter']
    Qe[elm, :] = args.IC(time, x)
    exactgradQ[elm,:,0]= 2*x[0]
    exactgradQ[elm,:,1]= 3
#--COMPUTE GRADIENT
# Instantinate gradient class
grd = grad(msh);  
# Set gradient options
grd.set(args); 
# Obtain boundary data using BC function
Qb   = grd.createBfield(Qe)
# Compute the gradient
gradQ = grd.compute(Qe, Qb)
#-------------Compute your errors below---------------#
graderror= np.abs(gradQ -exactgradQ)#calculate errors
infinitynorm=np.max(graderror)#calculate infinity norm
avgnorm=np.mean(graderror)
#Plotting
x_coords=[]
y_coords=[]
for elm, info in msh.Element.items():
    x = info['ecenter']
    x_coords.append(x[0])
    y_coords.append(x[1])
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
plt.quiver(x_coords, y_coords, gradQ[:,:,0], gradQ[:,:,1])
plt.xlim([min(x_coords), max(x_coords)])
plt.ylim([min(y_coords), max(y_coords)])
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f'Gradient Computation Method: {args.method} without correction')

# plt.quiver(x_coords, y_coords, exactgradQ[:,:,0], gradQ[:,:,1])
# plt.xlim([min(x_coords), max(x_coords)])
# plt.ylim([min(y_coords), max(y_coords)])
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Gradient Computation Method: Exact Solution')





