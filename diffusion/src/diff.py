import numpy as np
import scipy as sp
import argparse
from mesh import mesh
from .base import base
from .grad import grad

import matplotlib.pyplot as plt


import inspect
import pyamg

class diff():

#-------------------------------------------------------------------------------------------------#
    def __init__(self, mesh):
       
        self.mesh = mesh
        self.grd = grad(mesh)
#-------------------------------------------------------------------------------------------------#
    def set(self, args):
        self.args = args;
        self.Nfields = args.Nfields
        self.method  = args.method

        parser = argparse.ArgumentParser(prog='Gradient',
                    description='Compute Gradient using FVM',
                    epilog='--------------------------------')

        parser.add_argument('--method', type=str, default='WEIGHTED-LEAST-SQUARES')
        parser.add_argument('--Correct', type=bool, default=False)
        parser.add_argument('--Nfields', type=int, default=args.Nfields)
        parser.add_argument('--dim', type=int, default=args.dim)
        parser.add_argument('--IC', type = args.IC, default = args.IC)
        parser.add_argument('--BC', type = args.BC, default = args.BC)
        gargs = parser.parse_args()

        self.grd.set(gargs)
        self.bcFunc = args.BC

        # Set diffusion coefficient
        self.Ke = self.setFromFunction(args.DC)
        # Set source term
        self.Qs = self.setFromFunction(args.DS)    
#-------------------------------------------------------------------------------------------------#
    def assemble(self, Qe, Qb):
        # no orthogonal correction
        if(self.method=='NO-CORRECTION'):
            # Complete this function
            A, b = self.assembleOrthogonal(Qe, Qb)
        # else:
            # You do not need to implement following function
            # A, b = self.assembleCorrection(Qe, Qb)  
        return A, b  
 #-------------------------------------------------------------------------------------------------#
    def createFaceBfield(self, Qe):
        BCField =  np.zeros((self.mesh.NBFaces, self.Nfields), float)
        for face, info in self.mesh.Face.items():
            bc    = info['boundary']
            coord = info['center']
            if(bc !=0):
                bcid = info['bcid']
                eM   = info['owner']
                BCField[bcid]   = self.bcFunc(bc, 0.0, coord, Qe[eM])
        return BCField
#-------------------------------------------------------------------------------------------------#
    def createVertexBfield(self, Qe):
        BCField =  np.zeros((self.mesh.NBVertices, self.Nfields), float)
        for vrtx, info in self.mesh.Node.items():
            bc    = info['boundary']
            coord = info['coord']
            if(bc):
                bcid = info['bcid']
                BCField[bcid]   = self.bcFunc(bc, 0.0, coord, 0.0)
        return BCField
 
 #-------------------------------------------------------------------------------------------------#
    def setFromFunction(self, func):
        val = np.zeros((self.mesh.Nelements, self.Nfields), float)
        for elm, info in self.mesh.Element.items():
            xM      = info['ecenter']
            val[elm][:] = func(0.0, xM)

        return val

 #-------------------------------------------------------------------------------------------------#
    def extractBfield(self, Qe):
        BCField  = np.zeros((self.mesh.NBFaces, self.Nfields), float)
        for face, info in self.mesh.Face.items():
            bc    = info['boundary']
            coord = info['center']
            if(bc !=0):
                bcid = info['bcid']
                eM   = info['owner']
                # correct this later AK!
                if(self.mesh.BCMap[bc]['dtype'] == 'NEUMANN'):
                    BCField[bcid] = Qe[eM]
                else:
                    BCField[bcid]  = self.bcFunc(bc, 0.0, coord, Qe[eM])
        return BCField
#-------------------------------------------------------------------------------------------------#
    def assembleOrthogonal(self, Qe, Qb):
        # copy mesh class
        msh = self.mesh
        # Create dummy memory to hold indices and values of the sparse matrix
        # A(i,j) = val, i goes to "rows", j goes to "cols"
        # Note that we don't know the exact number of nonzero elements so give a large value (5)
        rows = np.zeros((5*msh.Nelements), int) -1
        cols = np.zeros((5*msh.Nelements), int) -1
        vals = np.zeros((5*msh.Nelements), float) -1
        # RHS vector (b in the homework) is not sparse
        rhs  = np.zeros((msh.Nelements,1), float)
        # This holds the number of non-zero entries, 
        # whenever you add an entry increase sk by one, i.e. sk = sk+1
        Ab=0
        Af=0
        sk = 0
        ctr = 0
        k=self.Ke
        Qs=self.Qs
        for elm, info in msh.Element.items():
            etype   = info['elementType']
            nfaces  = msh.elmInfo[etype]['nfaces'] 
            xM = info['ecenter']
            vol = info['volume']#Get Volume Info         
            rows[sk] = elm #Assign Ac location
            cols[sk] = elm #Assign Ac location
            vals[sk] = 0 #Assign Ac=0
            sk = sk+1 #Increase entry
            for face in range(nfaces):#Faces of element[elm]
                area = info['area'][face]#Get face area info
                bc = info['boundary'][face]#Get boundary info
                eP = info['neighElement'][face]#Get neighbor element info
                xP = msh.Element[eP]['ecenter']#Get neign elm coords
                Sf = area
                rows[sk] = elm#Assign center info
                cols[sk] = eP#Assign neigh info
                diff = xP-xM#Compute coords difference
                magdiff=np.linalg.norm(diff)
                if(bc!=0):#Drichlett Condition
                    xP = info['fcenter'][face]#Coords of boundary face center
                    diff = xP-xM#Compute Coords diff btw Boundary and elm center
                    magdiff=np.linalg.norm(diff)
                    Ab = k[elm]*Sf/magdiff
                    vals[sk-face-1] = vals[sk-face-1]+ Ab# Ac = Ac + Ab
                    vals[sk] = 0#Assign Af = 0 since boundary
                    sk = sk+1
                    rhs[elm] = rhs[elm]-(-Ab*Qb[ctr])# b = b - (-Ab*Qb)
                    ctr = ctr+1
                else:
                    Af = -k[elm]*Sf/magdiff
                    vals[sk] = Af # Assign Af to Af
                    vals[sk-face-1] = vals[sk-face-1]- Af# Ac = Ac + (-Af)
                    sk = sk+1
            rhs[elm] = -1*((Qs[elm]*vol) + rhs[elm])# -b = -(Qs*Vol + b)
        # Delete the unused memory
        rows = rows[0:sk]
        cols = cols[0:sk]
        vals = vals[0:sk]
        # Convert (i,j,val) tuple to sparse matrix
        A   = sp.sparse.coo_matrix((vals[:], (rows[:], cols[:])), 
            shape=(msh.Nelements, msh.Nelements), dtype=float)
        return A, rhs; 
#-------------------------------------------------------------------------------------------------#
    def assembleCorrection(self, Qe, Qb):
        # DO NOT COMPLETE THIS FUNCTION

        return A,b

#-------------------------------------------------------------------------------------------------#
    def solve(self, args, A, b):
        #----------------------------------------------------------------#
        # Define a callback function for solvers so that we can keep residuals
        res= []; iters = 0; 
        def report(xk):
            nonlocal iters
            frame = inspect.currentframe().f_back
            res.append(frame.f_locals['resid'])
            iters = iters+1

        msh = self.mesh
        #----------------------------------------------------------------#
        # Define Preconditioners for the solvers
        if(args.linPrecond == 'JACOBI'):
            A = sp.sparse.csr_matrix(A)
            M = sp.sparse.spdiags(1 / (A.diagonal()), 0, A.shape[0], A.shape[0])
        elif(args.linPrecond=='ILU'):
            A = sp.sparse.csc_matrix(A)
            sA_iLU = sp.sparse.linalg.spilu(A)
            M      = sp.sparse.linalg.LinearOperator((msh.Nelements, msh.Nelements), sA_iLU.solve)
        elif(args.linPrecond=='AMG'):
            A = sp.sparse.csr_matrix(A)
            Ml = pyamg.aggregation.smoothed_aggregation_solver(A)
            M =  Ml.aspreconditioner(cycle='V')        
        else:
            M = None
        #----------------------------------------------------------------#
        TOL = args.linTolerance
        X = np.zeros((self.mesh.Nelements,1), float)
        
        # Solve the problems
        if(args.linSolver=='DIRECT'):
            X[:,0] = sp.sparse.linalg.spsolve(A, b)
        else:
            if(args.linSolver == 'CG'):   
                X[:,0], info = sp.sparse.linalg.cg(A, b, M=M, tol = TOL, callback=report)
            elif(args.linSolver == 'GMRES'):
                X[:,0], info = sp.sparse.linalg.gmres(A, b, M=M, tol = TOL, callback=report)

            print(args.linSolver, 'is converged in', iters,'iterations with tolerance of', TOL, 
                'using', args.linPrecond, 'preconditioner')
            #----------------------------------------------------------------#
            # plot the residual in semilog scale
            plt.semilogy(res, color='k', linestyle ='dashed', 
                marker='o', markeredgecolor='r', markerfacecolor='b', linewidth= 2); 
            plt.xlabel('Iteration Number'); 
            plt.ylabel('Residual') 
            plt.title(args.linSolver + ' with '+ args.linPrecond + ' Preconditioner') 
            plt.show()
        return X

