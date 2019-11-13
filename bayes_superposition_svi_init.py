import os, sys
import random
import ntpath
from collections import defaultdict
import ntpath
import pandas as pd  # Windows users: I copy-paste the pandas,dateutil and pyltz folders from anaconda 2!! into the site-packages folder of pymol(only for temporary use, other wise it gets confused with the paths of the packages)
import numpy as np
from pandas import Series
# Biopython
from Bio import SeqRecord, Alphabet, SeqIO
from Bio.Seq import Seq
import Bio.PDB as PDB
from Bio.Seq import MutableSeq
from Bio.PDB.Polypeptide import is_aa
from Bio.SVDSuperimposer import SVDSuperimposer
# #Jax
# import jax.numpy as np
# import jax.random as random
# from jax.config import config as jax_config
# from jax.scipy.special import logsumexp
#Pymol
import pymol
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
# TORCH: "Tensors"
import torch
from torch.distributions import constraints, transform_to
#import tensorflow as tf
# PYRO
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal, AutoLowRankMultivariateNormal, init_to_median
from torch.optim import Adam, LBFGS
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO, TraceGraph_ELBO,JitTrace_ELBO
#Seaborn
import seaborn as sns
# Matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg') #TkAgg
import tqdm
tqdm.monitor_interval = 0
# Early STOPPING
from ignite.handlers import EarlyStopping
from ignite.engine import Engine, Events
from pyro.infer import SVI, EmpiricalMarginal
from pyro.optim import PyroOptim
# NUTS sampler
from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.util import initialize_model, predictive
from pyro.util import ignore_experimental_warning
# Posterior probabilities
from pyro.infer.abstract_infer import EmpiricalMarginal
#LOGGING (stats for posterior)
import logging

torch.multiprocessing.set_sharing_strategy('file_system')  # Needed to avoid runtime errors

class SVIEngine(Engine):
    def __init__(self, *args, step_args=None, **kwargs):
        self.svi = SVI(*args, **kwargs)
        self._step_args = step_args or {}
        super(SVIEngine, self).__init__(self._update)
    def _update(self, engine, batch):
        return -engine.svi.step(batch, **self._step_args)

PyroOptim.state_dict = lambda self: self.get_state()


def Max_variance(structure):
    '''Calculates the maximum distance to the origin of the structure, this value will define the variance of the prior's distribution'''
    centered = Center_torch(structure)
    mul = centered@torch.t(structure)
    max_var = torch.sqrt(torch.max(torch.diag(mul)))
    return max_var

def Extract_coordinates_from_PDB(PDB_file,type):
    ''' Returns both the alpha carbon coordinates contained in the PDB file and the residues coordinates for the desired chains'''
    from Bio.PDB.PDBParser import PDBParser
    from Bio.PDB import MMCIFParser
    Name = ntpath.basename(PDB_file).split('.')[0]

    try:
        parser = PDB.PDBParser()
        structure = parser.get_structure('%s' % (Name), PDB_file)
    except:
        parser = MMCIFParser()
        structure = parser.get_structure('%s' % (Name), PDB_file)

    ############## Iterating over residues to extract all of them even if there is more than 1 chain
    if type=='models':
        CoordinatesPerModel = []
        for model in structure:
            model_coord =[]
            for chain in model:
                for residue in chain:
                    if is_aa(residue.get_resname(), standard=True):
                            model_coord.append(residue['CA'].get_coord())
            CoordinatesPerModel.append(model_coord)

        return CoordinatesPerModel

    elif type=='chains':
        CoordinatesPerChain=[]
        for model in structure:
            for chain in model:
                chain_coord = []
                for residue in chain:
                    if is_aa(residue.get_resname(), standard=True):
                        chain_coord.append(residue['CA'].get_coord())
                CoordinatesPerChain.append(chain_coord)
        return CoordinatesPerChain

    elif type =='all':
        alpha_carbon_coordinates = []
        for chain in structure.get_chains():
            for residue in chain:
                if is_aa(residue.get_resname(), standard=True):
                    # try:
                    alpha_carbon_coordinates.append(residue['CA'].get_coord())
                # except:
                # pass
        return alpha_carbon_coordinates

def Average_Structure(tuple_struct):
    average = sum(list(tuple_struct)) / len(
        tuple_struct)  # sum element-wise the list of tensors containing the coordinates #tf.add_n
    return average

def Center_numpy(Array):
    '''Centering to the origin the data'''
    mean = np.mean(Array, axis=0)
    centered_array = Array - mean
    return centered_array

def Center_torch(Array):
    '''Centering to the origin the data'''
    mean = torch.mean(Array, dim=0)
    centered_array = Array - mean
    return centered_array

def Quaternions2Rotation(ri_vec):
    """Inputs a sample of unit quaternion and transforms it into a rotation matrix"""
    # argument i guarantees that the symbolic variable name will be identical everytime this method is called
    # repeating a symbolic variable name in a model will throw an error
    # the first argument states that i will be the name of the rotation made
    theta1 = 2 * np.pi * ri_vec[1]
    theta2 = 2 * np.pi * ri_vec[2]

    r1 = torch.sqrt(1 - ri_vec[0])
    r2 = torch.sqrt(ri_vec[0])

    qw = r2 * torch.cos(theta2)
    qx = r1 * torch.sin(theta1)
    qy = r1 * torch.cos(theta1)
    qz = r2 * torch.sin(theta2)

    R = torch.eye(3, 3) # device =cuda
    # filling the rotation matrix
    # Evangelos A. Coutsias, et al "Using quaternions to calculate RMSD" In: Journal of Computational Chemistry 25.15 (2004)

    # Row one
    R[0, 0] = qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2
    R[0, 1] = 2 * (qx * qy - qw * qz)
    R[0, 2] = 2 * (qx * qz + qw * qy)

    # Row two
    R[1, 0] = 2 * (qx * qy + qw * qz)
    R[1, 1] = qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2
    R[1, 2] = 2 * (qy * qz - qw * qx)

    # Row three
    R[2, 0] = 2 * (qx * qz - qw * qy)
    R[2, 1] = 2 * (qy * qz + qw * qx)
    R[2, 2] = qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2
    return R

def RMSD_numpy(X1, X2):
    import torch.nn.functional as F
    return F.pairwise_distance(torch.from_numpy(X1), torch.from_numpy(X2))

def RMSD(X1, X2):
    import torch.nn.functional as F
    return F.pairwise_distance(X1, X2)

def RMSD_biopython(x, y):
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rot, tran = sup.get_rotran()
    return rot

def Read_Data(prot1, prot2, type='models', models =(0,1), RMSD=True):
    '''Reads different types of proteins and extracts the alpha carbons from the models, chains or all . The model,
    chain or aminoacid range numbers are indicated by the tuple models'''

    if type == 'models':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]]

    elif type == 'chains':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]][0:141]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[1]][0:141]

    elif type == 'all':
        X1_coordinates = Extract_coordinates_from_PDB('{}'.format(prot1),type)[models[0]:models[1]]
        X2_coordinates = Extract_coordinates_from_PDB('{}'.format(prot2),type)[models[0]:models[1]]

    #Apply RMSD to the protein that needs to be superimposed
    X1_Obs_Stacked = Center_numpy(np.vstack(X1_coordinates))
    X2_Obs_Stacked = Center_numpy(np.vstack(X2_coordinates))

    if RMSD:
        X2_Obs_Stacked = torch.from_numpy(np.dot(X2_Obs_Stacked, RMSD_biopython(X1_Obs_Stacked, X2_Obs_Stacked)))
        X1_Obs_Stacked = torch.from_numpy(X1_Obs_Stacked)

    else:
        X1_Obs_Stacked = torch.from_numpy(X1_Obs_Stacked)
        X2_Obs_Stacked = torch.from_numpy(X2_Obs_Stacked)

    data_obs = (X1_Obs_Stacked, X2_Obs_Stacked)

    # ###PLOT INPUT DATA################

    x = Center_numpy(np.vstack(X1_coordinates))[:, 0]
    y = Center_numpy(np.vstack(X1_coordinates))[:, 1]
    z = Center_numpy(np.vstack(X1_coordinates))[:, 2]
    fig = plt.figure(figsize=(18, 16), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(x, y, z)
    ax.plot(x, y, z,  c='b', label='data1', linewidth=3.0)

    #orange graph
    x2 = Center_numpy(np.vstack(X2_coordinates))[:, 0]
    y2=Center_numpy(np.vstack(X2_coordinates))[:, 1]
    z2=Center_numpy(np.vstack(X2_coordinates))[:, 2]
    ax.plot(x2, y2,z2, c='r', label='data2',linewidth=3.0)
    ax.legend()
    plt.savefig(r"Initial.png")
    plt.clf() #Clear the plot, otherwise it will give an error when plotting the loss
    plt.close()

    rmsd = RMSD_numpy(Center_numpy(np.vstack(X1_coordinates)),Center_numpy(np.vstack(X2_coordinates)))
    plt.plot(rmsd.numpy())
    plt.savefig("RMSD.png")
    return data_obs

def model(data):

    max_var, data1, data2 = data

    ### 1. prior over mean M
    M = pyro.sample("M", dist.StudentT(1,0, 3).expand_by([data1.size(0),data1.size(1)]).to_event(2))

    ### 2. Prior over variances for the normal distribution - HalfCauchy?
    U = pyro.sample("U", dist.HalfNormal(1).expand_by([data1.size(0)]).to_event(1))
    U =  U.reshape(data1.size(0),1).repeat(1,3).view(-1)  #Triplicate the rows for the subsequent mean calculation

    ## 3. prior over translations T_i: Sample translations for each of the x,y,z coordinates
    T2 = pyro.sample("T2", dist.Normal(0, 1).expand_by([3]).to_event(1))

    ## 4. prior over rotations R_i
    ri_vec = pyro.sample("ri_vec",dist.Uniform(0, 1).expand_by([3]).to_event(1))  # Uniform distribution

    R = Quaternions2Rotation(ri_vec)
    M_T1 = M
    M_R2_T2 = M @ R + T2

    # 5. Sampling from several Univariate Distributions (approximating the posterior distribution ):
    # The observations are conditionally independant given the U, which is sampled outside the loop
    # UNIVARIATE NORMALS
    with pyro.plate("plate_univariate", data1.size(0)*data1.size(1), dim=-1):
        pyro.sample("X1", dist.StudentT(1, M_T1.view(-1), U),obs=data1.view(-1))
        pyro.sample("X2", dist.StudentT(1, M_R2_T2.view(-1), U), obs=data2.view(-1))

def _get_initial_trace(data_obs, average):
    '''Initialize MCMC and NUTS'''

    #INITIALIZE PRIOR:
    def init_prior(site):
        if site["name"] == "ri_vec":
            return torch.tensor([0.9, 0.1, 0.9])
        elif site["name"] == "M":
            return average
        else:
            return init_to_median(site)

    # GUIDE
    global_guide = AutoDelta(model,init_loc_fn=init_prior)
    # OPTIMIZER
    optim = pyro.optim.AdagradRMSProp(dict()) #https://github.com/pyro-ppl/pyro/blob/58277184310ef76a62420a38c300e84cd12b88ad/pyro/optim/adagrad_rmsprop.py
    elbo = JitTrace_ELBO()
    # STOCHASTIC VARIATIONAL INFERENCE
    svi_engine = SVIEngine(model, global_guide, optim, loss=elbo)
    pbar = tqdm.tqdm()
    loss_list = []
    # INITIALIZING PRIOR : Changing in the first iteration the value for the prior in order to constrain the prior over the rotations
    #pyro.param("auto_ri_vec", torch.Tensor([0.9, 0.1, 0.9]),constraint=constraints.unit_interval)  # constraint = constraints.simplex doesn't work exactly
    # Initialize the Mean Structure (each coordinate separately): NO CONSTRAINTS!!!
    #pyro.param("auto_M",average)
    @svi_engine.on(Events.EPOCH_COMPLETED)
    def update_progress(svi_engine):
        pbar.update(1)
        loss_list.append(-svi_engine.state.output)
        pbar.set_description(
            "[epoch {}] avg train loss: {}".format(svi_engine.state.epoch, svi_engine.state.output))
    # HANDLER
    # CHANGE PATIENCE PARAMETER FOR MORE COMPLEX PROTEINS
    handler = EarlyStopping(patience=100, score_function=lambda eng: eng.state.output, trainer=svi_engine)
    # SVI
    svi_engine.add_event_handler(Events.EPOCH_COMPLETED, handler)
    svi_engine.run([data_obs], max_epochs=15000)
    return svi_engine.svi.exec_traces

def GetPosteriorNUTS(data_obs, name1, samples=1250, warmup = 250, chains = 1):

    average = Average_Structure(data_obs)

    # Running MCMC using NUTS as selected kernel
    nuts_kernel = NUTS(model, jit_compile=True, ignore_jit_warnings=True, max_tree_depth=5)

    # INITIALIZING PRIOR Changing in the first iteration the value for the prior in order to constrain the prior over the rotations
    # _get_initial_trace(data_obs, average)
    nuts_kernel.initial_trace = _get_initial_trace(data_obs, average)

    mcmc = MCMC(nuts_kernel, num_samples=samples, warmup_steps=warmup, num_chains=chains)
    mcmc.run(data_obs)

    max_var, data1, data2 = data_obs

    samples_posterior = mcmc.get_samples()

    with ignore_experimental_warning():
        posterior_predicted = predictive(model, samples_posterior, data_obs)

    # Posterior_predicted["X1"] has dimensions (1250, 213) -> (n_samples, n_cA* 3 coordinates)
    nrows1 = data1.shape[0]
    x1 = posterior_predicted["X1"][0,:]
    X1 = x1.numpy().reshape([nrows1,3])

    nrows2 = data2.shape[0]
    x2 = posterior_predicted["X2"][0, :]
    X2 = x2.numpy().reshape([nrows2, 3])

    # Get summary statistics from the mcmc run
    #mcmc.summary(0.5)

    # Rotation matrix posterior samples
    ri_post_samples = samples_posterior["ri_vec"]

    # Mean structure M output
    M_post_samples = samples_posterior["M"]
    M = M_post_samples.mean(dim=0)
    # M is a tensor with dimensions (n*AA, 3)

    # Variance per row posterior samples
    U_post_samples = samples_posterior["U"]

    # Translation T output
    T2_post_samples = samples_posterior["T2"]

    import matplotlib
    matplotlib.rcParams['legend.fontsize'] = 10

    #################PLOTS################################################

    fig = plt.figure(figsize=(18, 16), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    # blue graph
    x = X1[:, 0]
    y = X1[:, 1]
    z = X1[:, 2]

    ax.plot(x, y, z, c='b', label='X1', linewidth=3.0)

    # red graph
    x2 = X2[:, 0]
    y2 = X2[:, 1]
    z2 = X2[:, 2]

    ax.plot(x2, y2, z2, c='r', label='X2', linewidth=3.0)

    ###green graph
    x3 = M.numpy()[:, 0]
    y3 = M.numpy()[:, 1]
    z3 = M.numpy()[:, 2]

    ax.plot(x3, y3, z3, c='g', label='Mean structure', linewidth=3.0)
    
    ax.legend()

    plt.title("Initialized NUTS model")
    plt.savefig("Bayesian_Result_Samples_{}_{}".format(name1, samples + warmup))

    plt.clf()
    plt.plot(RMSD(data1, data2).numpy(), linewidth = 8.0)
    plt.plot(RMSD(torch.from_numpy(X1), torch.from_numpy(X2)).numpy(), linewidth=8.0)
    plt.ylabel('Pairwise distances',fontsize='46')
    plt.xlabel('Amino acid position',fontsize='46')
    plt.title('{}'.format(name1.upper()),fontsize ='46')
    plt.gca().legend(('RMSD', 'NUTS Theseus-PP'),fontsize='40')
    plt.savefig(r"Distance_Differences_{}_Bayesian".format(name1))
    plt.close()

    return X1, X2, ri_post_samples, M_post_samples, T2_post_samples



def write_ATOM_line(structure, file_name):

    import os
    """Transform coordinates to PDB file: Add intermediate coordinates to be able to visualize Mean structure in PyMOL"""
    expanded_structure = np.ones(shape=(2 * len(structure) - 1, 3))  # The expanded structure contains extra rows between the alpha carbons
    averagearray = np.zeros(shape=(len(structure) - 1, 3))  # should be of size len(structure) -1
    for index, row in enumerate(structure):
        if index != len(structure) and index != len(structure) - 1:
            averagearray[int(index)] = (structure[int(index)] + structure[int(index) + 1]) / 2
        else:
            pass

    # split the expanded structure in sets , where each set will be structure[0] + number*average
    # The even rows of the 'expanded structure' are simply the rows of the original structure
    expanded_structure[0::2] = structure
    expanded_structure[1::2] = averagearray
    structure = expanded_structure
    aa_name = "ALA"
    aa_type = "CA"
    if os.path.isfile(file_name):
        os.remove(file_name)
        for i in range(len(structure)):
            with open(file_name, 'a') as f:
                f.write(
                    "ATOM{:7d} {}   {} A{:4d}{:12.3f}{:8.3f}{:8.3f}  0.00  0.00    X    \n".format(i, aa_type, aa_name,
                                                                                                   i, structure[i, 0],
                                                                                                   structure[i, 1],
                                                                                                   structure[i, 2]))

    else:
        for i in range(len(structure)):
            with open(file_name, 'a') as f:
                f.write(
                    "ATOM{:7d} {}   {} A{:4d}{:12.3f}{:8.3f}{:8.3f}  0.00  0.00    X    \n".format(i, aa_type, aa_name,
                                                                                                   i, structure[i, 0],
                                                                                                   structure[i, 1],
                                                                                                   structure[i, 2]))


def Pymol(*args):
    '''Visualization program'''

    #LAUNCH PYMOL
    launch=False
    if launch:
        pymol.pymol_argv = ['pymol'] + sys.argv[1:]
        pymol.finish_launching(['pymol'])
    def Colour_Backbone(selection,color,color_digit):
        #pymol.cmd.select("alphas", "name ca") #apparently nothing is ca
        #pymol.cmd.select("sidechains", "! alphas") #select the opposite from ca, which should be the side chains, not working
        pymol.cmd.show("sticks", selection)
        pymol.cmd.set_color(color,color_digit)
        pymol.cmd.color(color,selection)

    # Load Structures and apply the function
    #colornames=['red','green','blue','orange','purple','yellow','black','aquamarine']
    #Palette of colours
    pal = sns.color_palette("PuBuGn_d",100) #RGB numbers for the palette colours
    colornames = ["blue_{}".format(i) for i in range(0,len(pal))]
    snames=[]
    for file,color,color_digit in zip(args,colornames,pal):
        sname = ntpath.basename(file)
        snames.append(sname)
        pymol.cmd.load(file, sname) #discrete 1 will create different sets of atoms for each model
        pymol.cmd.bg_color("white")
        pymol.cmd.extend("Colour_Backbone", Colour_Backbone)
        Colour_Backbone(sname,color,color_digit)
    pymol.cmd.png("Superposition_Bayesian_Pymol_{}".format(snames[0].split('_')[2]))



def Sample_Posterior(X1, X2, name1, ri_MCMCsamples, T_MCMCsamples, num_samples):

    # Choose a random sample between 0 and the number of MCMC samplings, the number of times specified in num_samples
    indexes = random.sample(range(0, 1250), num_samples)

    plt.clf()

    for i in indexes:

        Rotation = Quaternions2Rotation(ri_MCMCsamples[i,:]) #torch
        invRotation = Rotation.inverse()
        Translation = T_MCMCsamples[i,:]
        X2_NUTS = np.dot(X2.numpy() + Translation.numpy(), invRotation.cpu().numpy())
        write_ATOM_line(X2_NUTS, os.path.join("{}_PDB_files".format(name1),'NUTS_{}_X2_{}.pdb'.format(name1, i)))


        plt.plot(RMSD(X1, torch.from_numpy(X2_NUTS)).numpy(), linewidth=1.0, c="b", alpha=0.05)

    plt.plot(RMSD(X1, X2).numpy(), linewidth=3.0, c="r")
    plt.ylabel('Pairwise distances (Angstroms)')
    plt.xlabel('Amino acid position')
    plt.title('{}'.format(name1.upper()))
    plt.savefig(r"Distance_NUTS_{}".format(name1))
    plt.close()


    #names = [os.path.join("{}_PDB_files".format(name1),'NUTS_{}_X2_{}.pdb'.format(name1, i)) for i in indexes] #exchange indexes with range(0, num_samples)
    #Pymol(*names)



def Create_Folder(folder_name):
    """ Folder for all the generated images It will updated everytime!!! Save the previous folder before running again. Creates folder in current directory"""
    import os
    import shutil
    basepath = os.getcwd()
    if not basepath:
        newpath = folder_name
    else:
        newpath = basepath + "/%s" % folder_name

    if not os.path.exists(newpath):
        try:
            original_umask = os.umask(0)
            os.makedirs(newpath, 0o777)
        finally:
            os.umask(original_umask)
    else:
        shutil.rmtree(newpath)  # removes all the subdirectories!
        os.makedirs(newpath,0o777)



#if __name__ == "__main__":
name1 ="my_3qrf"
name2 ="my_1owr"
Create_Folder("{}_PDB_files".format(name1))

# Read protein data for X1 and X2
data_obs = Read_Data('/home/pabswfly/PycharmProjects/theseus/PDB_files/{}.pdb'.format(name1),
                     '/home/pabswfly/PycharmProjects/theseus/PDB_files/{}.pdb'.format(name2),
                     type='all', models =(0, 100), RMSD=True)

# Organize protein data
max_var = Max_variance(data_obs[0])
data1, data2 = data_obs
data_obs = max_var, data1, data2

# Get samples from the posterior distribution inferred with MCMC and NUTS
X1, X2, ri_post_samples, M_post_samples, T_post_samples = GetPosteriorNUTS(data_obs, name1)

# Write PDBs for prot1 and prot2 superimposed with RMSD
write_ATOM_line(data1, os.path.join("{}_PDB_files".format(name1),'RMSD_{}_data1.pdb'.format(name1)))
write_ATOM_line(data2, os.path.join("{}_PDB_files".format(name1),'RMSD_{}_data2.pdb'.format(name1)))

# Write PDBs for X1 and X2 superimposed with NUTS inference
write_ATOM_line(X1, os.path.join("{}_PDB_files".format(name1),'Result_MCMC_{}_X1.pdb'.format(name1)))
write_ATOM_line(X2, os.path.join("{}_PDB_files".format(name1),'Result_MCMC_{}_X2.pdb'.format(name2)))

#Pymol("Result_MCMC_{}_X1.pdb".format(name1), "Result_MCMC_{}_X2.pdb".format(name2))
num_samples = 20
Sample_Posterior(data1, data2, name1, ri_post_samples, T_post_samples, num_samples)




















