#!/usr/bin/env python
# This is a modified version of example7.py of LFPy

import LFPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import neuron
import math
import io
import sys

plt.rcParams.update({'font.size' : 12,
                     'figure.facecolor' : '1',
                     'figure.subplot.wspace' : 0.5,
                     'figure.subplot.hspace' : 0.5})

np.random.seed(12345)

dryrun = False
L = 300
centerpos = (0, 0, 0)
themode = sys.argv[1]
if themode not in ['msn', 'msn-beta', 'pyramidal', 'pyramidal-beta']:
    sys.exit("Please specify the mode (msn, msn-beta, pyramidal, or pyramidal-beta).")

radius = 150
synapticweight = 0.001
if themode in ['msn', 'msn-beta']: # MSN
    cortex = False
    Ncells = 2700*8
    gabaradius = 200
    gabafactor = 1.5
    synapsePerCell = 80
elif themode in ['pyramidal', 'pyramidal-beta']: # Cortex
    cortex = True
    Ncells = 2700*60/100*8
    gabaradius = 1e5
    gabafactor = 0.1
    synapsePerCell = 200
if themode in ['msn', 'pyramidal']:
    tstopms = 100
elif themode in ['msn-beta', 'pyramidal-beta']:
    tstopms = 200

optionstr = themode+'_newmodel'

def insert_synapses(centerpos, radius, dryrun, synparams, section, n, spTimesFun, args):
    if section is None:
        idx = cell.get_rand_idx_area_norm(nidx=n)
    else:
        idx = cell.get_rand_idx_area_norm(nidx=n, section=section)

    inserted = 0
    for i in idx:
        if (cell.xmid[i]-centerpos[0])**2+(cell.ymid[i]-centerpos[1])**2+(cell.zmid[i]-centerpos[2])**2<=radius**2:
            inserted += 1
            if not dryrun:
                synparams.update({'idx' : int(i)})
                spiketimes = spTimesFun(args[0], args[1], args[2], args[3], args[4])
                s = LFPy.Synapse(cell, **synparams)
                s.set_spike_times(spiketimes)
    return inserted

def nonstationary_poisson(tstart, tstop, maxlambda, frequency, tmin=-1000.0, tmax=1000000.0):
    X = []
    x = tmin
    while True:
        x += -np.log(np.random.random())*maxlambda
        if x>tstop:
            break
        if np.random.random()<(1+np.sin(-np.pi/2+2*np.pi*frequency*(x-tstart)/1000.0))/2:
            X.append(x)
    return np.array(X)

if cortex:
    cellParameters = {
        'morphology' : 'morphologies/L5_Mainen96_wAxon_LFPy.hoc',
        'rm' : 30000,               # membrane resistance
        'cm' : 1.0,                 # membrane capacitance
        'Ra' : 150,                 # axial resistance
        'v_init' : -65,             # initial crossmembrane potential
        'e_pas' : -65,              # reversal potential passive mechs
        'passive' : True,           # switch on passive mechs
        'nsegs_method' : 'lambda_f',# method for setting number of segments,
        'lambda_f' : 100,           # segments are isopotential at this frequency
        'timeres_NEURON' : 2**-4,   # dt of LFP and NEURON simulation.
        'timeres_python' : 2**-4,
        'tstartms' : -50,          #start time, recorders start at t=0
        'tstopms' : tstopms,            #stop time of simulation
    }
else:
    cellParameters = {
        'morphology' : 'morphologies/msp_template_modified2.hoc',
        'rm' : 1/(1.7e-5*1.3),               # membrane resistance
        'cm' : 1.0,                 # membrane capacitance
        'Ra' : 100,                 # axial resistance
        'v_init' : -70,             # initial crossmembrane potential
        'e_pas' : -70,              # reversal potential passive mechs
        'passive' : True,           # switch on passive mechs
        'nsegs_method' : 'lambda_f',# method for setting number of segments,
        'lambda_f' : 100,           # segments are isopotential at this frequency
        'timeres_NEURON' : 2**-4,   # dt of LFP and NEURON simulation.
        'timeres_python' : 2**-4,
        'tstartms' : -50,          #start time, recorders start at t=0
        'tstopms' : tstopms,            #stop time of simulation
    }

# Synaptic parameters taken from Hendrickson et al 2011
# Excitatory synapse parameters:
synapseParameters_AMPA = {
    'e' : 0,                    #reversal potential
    'syntype' : 'Exp2Syn',      #conductance based exponential synapse
    'tau1' : 1.,                #Time constant, rise
    'tau2' : 3.,                #Time constant, decay
    'weight' : synapticweight,           #Synaptic weight
    'color' : 'r',              #for plt.plot
    'marker' : '.',             #for plt.plot
    'record_current' : True,    #record synaptic currents
}

# Inhibitory synapse parameters
synapseParameters_GABA_A = {         
    'e' : -70,
    'syntype' : 'Exp2Syn',
    'tau1' : 1.,
    'tau2' : 12.,
    'weight' : synapticweight,
    'color' : 'b',
    'marker' : '.',
    'record_current' : True
}

# where to insert, how many, and which input statistics
if themode in ['msn', 'pyramidal']:
    insert_synapses_AMPA_args = {
        'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
        'args' : [25, 75, 0.5, 40,
                  cellParameters['tstartms']]
        }

    insert_synapses_GABA_A_args = {
        'spTimesFun' : LFPy.inputgenerators.stationary_gamma,
        'args' : [25, 75, 0.5, 40,
                  cellParameters['tstartms']]
        }
else:
    insert_synapses_AMPA_args = {
        'spTimesFun' : nonstationary_poisson,
        'args' : [0, tstopms, 20, 15,
                  cellParameters['tstartms']]
        }

    insert_synapses_GABA_A_args = {
        'spTimesFun' : nonstationary_poisson,
        'args' : [0, tstopms, 20, 15,
                  cellParameters['tstartms']]
        }

# Define electrode geometry corresponding to a laminar electrode, where contact
# points have a radius r, surface normal vectors N, and LFP calculated as the
# average LFP in n random points on each contact:
n_elec = 64
N = np.empty((n_elec, 3))
for i in range(N.shape[0]):
    N[i,] = [1, 0, 0] #normal unit vec. to contacts
if cortex:
    electrodeParameters = {
        'sigma' : 0.3,              # Extracellular potential
        'x' : np.zeros(n_elec) + 25*0,      # x,y,z-coordinates of electrode contacts
        'y' : np.zeros(n_elec),
        'z' : np.linspace(-1500, 500, n_elec),
        'n' : 20,
        'r' : 10,
        'N' : N,
    }
else:
    electrodeParameters = {
        'sigma' : 0.3,              # Extracellular potential
        'x' : np.zeros(n_elec) + 25*0,      # x,y,z-coordinates of electrode contacts
        'y' : np.zeros(n_elec),
        'z' : np.linspace(-500, 500, n_elec),
        'n' : 20,
        'r' : 10,
        'N' : N,
    }

# Parameters for the cell.simulate() call, recording membrane- and syn.-currents
simulationParameters = {
    'rec_imem' : True,  # Record Membrane currents during simulation
    'rec_isyn' : True,  # Record synaptic currents
    'rec_vmem' : False
}

################################################################################
# Main simulation procedure
################################################################################

total_ampa = 0
total_gaba = 0
totalneurons = 0
LFPsum = np.array([])
spos = []

print str(Ncells/float((2*L)**3)*100**3)+" neurons in 100 * 100 * 100 micro m^3"

fig = plt.figure(figsize=[12, 8])
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{arev}')
    
#plot the somatic trace
ax1 = fig.add_axes([0.4, 0.7, 0.5, 0.2])
ax1.set_xlabel('Time/ms')
ax1.set_ylabel('Soma potential/mV')

ax2 = fig.add_axes([0.4, 0.4, 0.5, 0.2])
ax2.set_xlabel('Time/ms')
ax2.set_ylabel('Synaptic input/nA')
    
#plot the morphology, electrode contacts and synapses
ax4 = fig.add_axes([0.05, 0, 0.25, 1], frameon=False)
ax4.axis('equal')

for nc in range(Ncells):
    #Initialize cell instance, using the LFPy.Cell class
    cell = LFPy.Cell(**cellParameters)
    
    if cortex:
        cell.set_rotation(0, 0, 2*math.pi*np.random.rand())
        cell.set_pos(-L+2*L*np.random.rand(), -L+2*L*np.random.rand(), -800-L+2*L*np.random.rand())
        ampa = insert_synapses(centerpos, radius, dryrun, synapseParameters_AMPA, 'apic', synapsePerCell, **insert_synapses_AMPA_args)
        gaba = insert_synapses(centerpos, gabaradius, dryrun, synapseParameters_GABA_A, 'dend', synapsePerCell*gabafactor, **insert_synapses_GABA_A_args)
    else:
        for i in range(10):
            cell.set_rotation(2*math.pi*np.random.rand(), 2*math.pi*np.random.rand(), 2*math.pi*np.random.rand())
        cell.set_pos(-L+2*L*np.random.rand(), -L+2*L*np.random.rand(), -L+2*L*np.random.rand())
        ampa = insert_synapses(centerpos, radius, dryrun, synapseParameters_AMPA, None, synapsePerCell, **insert_synapses_AMPA_args)
        gaba = insert_synapses(centerpos, gabaradius, dryrun, synapseParameters_GABA_A, None, synapsePerCell*gabafactor, **insert_synapses_GABA_A_args)
            
    #Insert synapses using the function defined earlier
    if ampa+gaba==0:
        continue
    total_ampa += ampa
    total_gaba += gaba
    totalneurons += 1
    if dryrun:
        print('processing neurons.... '+str(nc+1)+'/'+str(Ncells))
    else:
        print('simulating LFPs.... '+str(nc+1)+'/'+str(Ncells))
        #perform NEURON simulation, results saved as attributes in the cell instance
        cell.simulate(**simulationParameters)
    
        # Initialize electrode geometry, then calculate the LFP, using the
        # LFPy.RecExtElectrode class. Note that now cell is given as input to electrode
        # and created after the NEURON simulations are finished
        electrode = LFPy.RecExtElectrode(cell, **electrodeParameters)
        electrode.calc_lfp()

        thetvec = cell.tvec

        if LFPsum.shape==(0,):
            LFPsum = electrode.LFP
        else:
            LFPsum += electrode.LFP

    if totalneurons%100==0:
        thecolor = (np.random.rand(), np.random.rand(), np.random.rand())
    
        if not dryrun:
            ax1.plot(cell.tvec, cell.somav)

        for i in range(len(cell.synapses)):
            if i%10==0:
                ax2.plot(cell.tvec, cell.synapses[i].i, color=thecolor)
    
        for sec in neuron.h.allsec():
            idx = cell.get_idx(sec.name())
            ax4.plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]], # xstart and xend are lists of segments
                 np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                 color=thecolor)
        for i in range(len(cell.synapses)):
            spos.append([cell.synapses[i].x, cell.synapses[i].z, cell.synapses[i].color, cell.synapses[i].marker])

print totalneurons, "neurons,", total_ampa, "AMPA synapses,", total_gaba, "GABA synapses, "
for x,z,c,m in spos:
    ax4.plot(x, z, color=c, marker=m)
if not dryrun:
    for i in range(electrode.x.size):
        ax4.plot(electrode.x[i], electrode.z[i], color='g', marker='o')
    #plot the LFP as image plot
    ax3 = fig.add_axes([0.4, 0.1, 0.5, 0.2])
    absmaxLFP = abs(np.array([LFPsum.max(), LFPsum.min()])).max()
    im = ax3.pcolormesh(thetvec, electrode.z, LFPsum,
                        vmax=absmaxLFP, vmin=-absmaxLFP,
                        #cmap='spectral_r')
                        cmap='RdBu')
    #cb = plt.colorbar(im)
    #ticklabs = cb.ax.get_yticklabels()
    #cb.ax.set_yticklabels(ticklabs, ha='right')
    #cb.ax.yaxis.set_tick_params(pad=1000)

    ax3.axis(ax3.axis('tight'))
    ax3.set_xlabel('Time/ms')
    ax3.set_ylabel('z/$\mu$m')
    rect = np.array(ax3.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.02
    cax = fig.add_axes(rect)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('LFP/mV')

#plt.axis('equal')
#plt.axis(np.array(plt.axis())*0.8)
ax4.set_xticks([])
ax4.set_yticks([])

params = '_'.join(list(map(str, [totalneurons, total_ampa, total_gaba, Ncells, synapticweight, gabaradius, gabafactor, synapsePerCell])))

fig.savefig(optionstr+'_'+params+'.pdf', dpi=1200)

fp = open(optionstr+'_'+params+'_voltage.txt', 'w')
for l in LFPsum:
    fp.write(" ".join(map(str, l))+"\n")
fp.close()
