Tanaka T and Nakamura KC.
"Focal inputs are a potential origin of local field potential (LFP) in the brain regions without laminar structure"
PLOS ONE (in press)

Current sinks and sources spatially separated between the apical and basal dendrites have been believed to be essential in generating local field potentials (LFPs). According to this theory, LFPs would not be large enough to be observed in the regions without laminar structures, such as striatum and thalamus. However, LFPs are experimentally recorded in these regions. We hypothesized that focal excitatory input induces a concentric current sink and source generating LFPs in these regions. In this study, we tested this hypothesis by the numerical simulations of multicompartment neuron models and the analysis of simplified models. Both confirmed that focal excitatory input can generate LFPs on the order of 0.1 mV in a region without laminar structures. The present results suggest that LFPs in subcortical nuclei indicate localized excitatory input.

The model is based on example7.py in LFPy 1.1.3 demo codes (https://lfpy.github.io/v1.3/).

To reproduce our results, run

> python passive3.py pyramidal (Fig 2, Fig 5)
> python passive3.py msn (Fig 3, Fig 5)
> python passive3.py pyramidal-beta (Fig 4)
> python passive3.py msn-beta (Fig 4)

with Python 2.7.

Each of these commands generates a PDF file, which is a figure like Fig 2 and Fig 3, and a text file containing the time series of LFPs recorded by extracellular electrodes, which can be used to reproduce Fig 4 and Fig 5.

The morphology file of a pyramidal neuron is L5_Mainen96_wAxon_LFPy.hoc, which is distributed as a part of LFPy 1.1.3. Its original file by Mainen and Sejnowski is available from ModelDB (https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=2488).

The morphology file of a medium-sized spiny neuron is msp_template_modified2.hoc, whose original file by Nakano, Yoshimoto, and Doya is available from ModelDB (https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=151458).
