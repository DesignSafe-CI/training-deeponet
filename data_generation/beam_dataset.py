# Script to generate the cantilever beam dataset using FEniCS X 
# Written by: Wei Wang, Researcher at Centrum Intelliphysics, Johns Hopkins University
# Problem statement: Record the deflection of a cantilever beam of length (L) 2 units and thickness (H) = 0.2 units
import os
import jax
import jax.numpy as jnp
from jax import grad, vmap, random, config
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import gmsh
from dolfinx.fem import functionspace, Function 
from dolfinx import mesh
from dolfinx import fem
import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import pyvista
import matplotlib.pyplot as plt
from dolfinx import plot
from dolfinx import io
from pathlib import Path
import pickle
from scipy.interpolate import Rbf, interp1d, griddata
#from DeepONet_3_bc_test import PI_DeepONet
import math 
from tqdm import trange
import pyvista
import os
'''
1. In fenicsx or Gmsh, it doesn't support to jnp. But in RBF, we use jnp. Should convert np  to jnp. 

'''


#region utils 
def createFolder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except OSError:
        print ('Error: Creating folder. ' +  folder_name)



def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def plot_disp(X, Y, u, foldername):
    plt.figure(figsize=(10, 2))
    plt.scatter(X, Y, c=u, cmap='viridis')
    plt.colorbar(label='u')
    plt.xlabel('X', fontsize=32)
    plt.ylabel('Y', fontsize=32)
    plt.tick_params(axis='both', labelsize=28)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)      
    ax.spines['right'].set_linewidth(2)    
    ax.spines['bottom'].set_linewidth(2)   
    ax.spines['left'].set_linewidth(2) 
    #plt.yscale('log')
    plt.legend(fontsize=24, frameon=False)
    plt.title(foldername  )
    plt.savefig(foldername + ".jpg", dpi=700)
    plt.show()

def plot_bc(bc, dis, filename):
    plt.figure(figsize = (6,5))
    plt.plot(bc,dis, lw=4, label=filename)
    plt.xlabel('x',fontsize=24)
    plt.ylabel('displament',fontsize=24)    
    plt.tick_params(axis='both', labelsize=20)
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)      
    ax.spines['right'].set_linewidth(1.5)    
    ax.spines['bottom'].set_linewidth(1.5)   
    ax.spines['left'].set_linewidth(1.5) 
    #plt.yscale('log')
    plt.legend(fontsize=24, frameon=False)
    plt.tight_layout()
    plt.savefig(filename + ".jpg", dpi=700)
    plt.show()
    plt.close()   
        
originalDir ='/nfsv4/21040463r/FEM_DeepONet'
os.chdir(os.path.join(originalDir))

foldername = 'cantilever_beam_dataset'  
createFolder(foldername )
os.chdir(os.path.join(originalDir, './'+ foldername + '/'))  

  
#region Gmsh 
lc = 0.02
L = 2.
H = 0.2
# GMSH
gmsh.initialize()
gmsh.model.add("model")

gmsh.model.occ.addPoint(0, 0, 0, lc, 1)
gmsh.model.occ.addPoint(L, 0., 0, lc, 2)
gmsh.model.occ.addPoint(L, H, 0, lc, 3)
gmsh.model.occ.addPoint(0, H, 0, lc, 4)

line1 = gmsh.model.occ.addLine(1, 2)
line2 = gmsh.model.occ.addLine(2, 3)
line3 = gmsh.model.occ.addLine(3, 4)
line4 = gmsh.model.occ.addLine(4, 1)

rec_loop = gmsh.model.occ.addCurveLoop([line1, line2, line3, line4])
surface = gmsh.model.occ.addPlaneSurface([rec_loop])

# Synchronize the model to GMSH
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)


gmsh.model.addPhysicalGroup(2, [surface], name="beam")

# Write the mesh to a file (optional)
gmsh.write("cantilever_beam.msh")

# Finalize GMSH
gmsh.finalize()


import meshio
msh = meshio.read("cantilever_beam.msh")
points = msh.points
cells = msh.cells_dict["triangle"]  
plt.figure(figsize=(8, 8))
for cell in cells:
    polygon = points[cell]
    polygon = np.vstack([polygon, polygon[0]])
    plt.plot(polygon[:, 0], polygon[:, 1], 'k-') 

'''
plt.gca().set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gmsh Mesh Visualization')
plt.savefig('Gmsh Mesh Visualization' + ".jpg", dpi=700)
plt.show()
'''

# region Myexpression 
class MyExpression:
    def __init__(self, x0, y0, value, V_dim):
        self.x0 = x0
        self.y0 = y0
        self.value  = value
        self.V_dim = V_dim        
        self.RBF_1  = Rbf(x0, y0, value)
        
    def eval(self, x):
        #print(x[0], x[1], x[2]) 
        values = np.zeros((self.V_dim, x.shape[1]))
        values[1] = np.where(np.isclose(x[1], H, 1e-4, 1e-4), self.RBF_1(x[0], x[1]), 0)
        idx = np.where(np.isclose(x[1], H, 1e-4, 1e-4))
        #print(values[1, idx])    
        #print(values[1].shape)
        #print(x[0, idx])
        plot_bc(x[0, idx].flatten(), values[1, idx].flatten(), 'u_c test2')
        return values 


# FEM & Params
import dolfinx 
mesh1, cell_markers, facet_markers  = gmshio.read_from_msh("cantilever_beam.msh", MPI.COMM_WORLD) 
V = functionspace(mesh1, ("CG", 1, (mesh1.geometry.dim, ))) ### without  (mesh1.geometry.dim, ), it is a scalar space not a vector space 

### mild steel material parameters
E = 210e9 #Pa
nu = 0.33 
mu = E/(2 * (1 + nu))
lambda_ = E*nu/((1 + nu)*(1 - 2*nu))
m = 100 # sensor points 
key = random.PRNGKey(0)
length_scale = 0.3
N = 1000 # number of samples
keys = random.split(key, N)

# region RGF    
# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params): #radial basis function 
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale**2 * np.exp(-0.5 * r2)

def Generate_bcs(key, L, length_scale, x_c):
    """No need explicit resolution 
    """

    # Generate subkeys
    subkeys= random.split(key, 2)
    # Generate a GP sample
    N = 512
    gp_params = (0.2, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, L, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*np.eye(N))
    
    def gp_sample(key, L):
        gp_sample = jnp.dot(L, random.normal(key, (N,)))
        return gp_sample
    
    gp_sample_y = vmap(gp_sample, (0,None))(subkeys, L)
    # Create a callable interpolation function
    f_fn_v = lambda x: vmap(jnp.interp,(None, None, 0))(x, X.flatten(), gp_sample_y)
    u_c, v_c = f_fn_v(x_c)

    return u_c, v_c

# region BCs 
tdim = mesh1.topology.dim
fdim = tdim - 1
domain = mesh1

left_b = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0))
uD = np.array([0, 0, 0], dtype=default_scalar_type)
domain.topology.create_connectivity(fdim, tdim)
boundary_dofs = fem.locate_dofs_topological(V, fdim, left_b)
bc_left = fem.dirichletbc(uD, boundary_dofs, V)

x_c = np.linspace(0, L, m)
y_c = np.ones_like(x_c) * H

# Generate N samples of Dirichlet BCs
# convert np to jnp
config.update("jax_enable_x64", True)
u_c, v_c = vmap(Generate_bcs, (0, None, None, None))(keys, L, length_scale, jnp.array(x_c))

config.update("jax_enable_x64", False)

# region satic solver
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
ds = ufl.Measure("ds", domain=domain)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

u_list = []
v_list = []
for i in trange(7):
    upper_b = dolfinx.mesh.locate_entities_boundary(mesh1, fdim, lambda x: np.isclose(x[1], 0.2 ,1e-4, 1e-4))
    uD_c = Function(V)
    uD_c_value = u_c[i,:]
    uD_c_fun = MyExpression(x_c, y_c, uD_c_value, mesh1.geometry.dim)
    uD_c.interpolate(uD_c_fun.eval)        
    boundary_dofs = fem.locate_dofs_topological(V, fdim, upper_b)
    bc_c = fem.dirichletbc(uD_c, boundary_dofs)  
          
    bcs  = [bc_left, bc_c]
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    #PETSc（Portable, Extensible Toolkit for Scientific Computation）
    uh = problem.solve()### first time, assume T at interface = 0 
    u_geometry = mesh1.geometry.x
    u_values = uh.x.array.real
    u_tot = u_values.reshape(-1,3)
    u, v= u_tot[:,0], u_tot[:,1]
    u_list.append(u.reshape(-1,1))
    v_list.append(v.reshape(-1,1))
    
    if i <= 6:
        # Start virtual framebuffer if off-screen rendering is needed
        pyvista.start_xvfb()

        # Create plotter and pyvista grid
        p = pyvista.Plotter(off_screen=True)
        topology, cell_types, geometry = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Attach vector values to grid and warp grid by vector
        grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))

        # Add the grid to the plotter
        actor_0 = p.add_mesh(grid, style="wireframe", color="k")
        warped = grid.warp_by_vector("u", factor=1)
        actor_1 = p.add_mesh(warped, show_edges=True)
        #actor_1 = p.add_mesh(warped, show_edges= False)
        p.show_axes()
        #p.add_mesh(grid, show_edges=True)

        # Adjust the view to align with the XY plane
        p.view_xy()

        # Show the plot (necessary for off-screen rendering)
        p.show()

        # Save the screenshot
        p.screenshot(f"deflection {i}.png")


print('Finished data generation. Saving the files.')

resultdir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(resultdir):
    os.makedirs(resultdir)

# region save data
np.savetxt(os.path.join(resultdir,'GRF_applied.txt'), np.transpose(u_c))
np.savetxt(os.path.join(resultdir,'sensor_loc.txt'), x_c)  
disp_x = np.transpose(np.array(u_list)[:,:,0])
disp_y = np.transpose(np.array(v_list)[:,:,0])
np.savetxt(os.path.join(resultdir,'disp_x.txt'), disp_x)
np.savetxt(os.path.join(resultdir,'disp_y.txt'), disp_y)
coord_x, coord_y = u_geometry[:,0], u_geometry[:,1]    
np.savetxt('./data/coord_x.txt', coord_x)
np.savetxt('./data/coord_y.txt', coord_y)
