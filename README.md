# tumor_np

Resources in this repository. 

/code
Core analytic pipeline for manuscript in form of python notebooks. Where relevant, notebooks are named with reference to pertinent figures. 

/data
Raw spiking data is shared for all 1,196 neurons in the database. Raw expression strength data is shared for all 154 assemblies in the database. There are two datastructures with the following format. 

1. neuron_data_structure.pkl
Each row is an individual neuron. Columns are as follows: 
- `subject_id` - subject index
- `insertion_index` - insertion index
- `region` - anatomical region of insertion, as determined by intraop stereotactic coordinate co-registration with preoperative MRI
- `flair` - 1 if insertion is confirmed in FLAIR+ cortex by co-registration process
- `grade` - tumor grade, as determined by final pathology report
- `path` - tumor path, as determined by final pathology report
- `depth` - depth from pial surface (NaN if theused deep probe montage and therefore has no depth data)
- `waveform_cluster` - combined (waveform + spiking metrics) K-means cluster id. . 0 = putative excitatory, 1 = positive waveform, 2 = putative inhibitory (NaN if the used deep probe montage and therefore has no depth data).
- `beh_tstat_prod` - production-activity behavioral t-statistic (NaN if the session has no speech production)
- `beh_tstat_rec` - reception-activity behavioral t-statistic (NaN if the session has no speech reception)
- `information_capacity` - information capacity (entropy - temporal autocorrelative mutual information)
- `spike_times` - raw full-recording spike times (seconds)

2. assembly_data_structure.pkl
Each row is an individual assembly. Columns are as follows: 
- `subject_id` - subject index
- `insertion_index` - insertion index
- `assembly_index` - assembly index (i.e. which assembly within that insertion)
- `region` - anatomical region of insertion, as determined by intraop stereotactic coordinate co-registration with preoperative MRI
- `flair` - 1 if insertion is confirmed in FLAIR+ cortex by co-registration process
- `grade` - tumor grade, as determined by final pathology report
- `path` - tumor path, as determined by final pathology report
- `weight_vector` - vector of length = number of neurons communicating the degree to which each neuron contributes to the assembly. Key output of the PCA/ICA assembly identification pipeline. 
- `beh_tstat_prod` - production-activity behavioral t-statistic (NaN if the session has no speech production)
- `beh_tstat_rec` - reception-activity behavioral t-statistic (NaN if the session has no speech reception)
- `information_capacity` - assembly information capacity (computed from the ICA component time series)
- `expression_strength` - raw expression-strength time series across the recording
