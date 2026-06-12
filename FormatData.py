import uproot
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
import os

outputDir = '../FormattedData/'
os.makedirs(outputDir, exist_ok=True)
recoFile = uproot.open("/exp/sbnd/data/users/castalyf/Omnifold_data/t2k-nd280-numu-cc0pi-training-sample/NuMuCC0pi_reco_v2.root")
truthFile = uproot.open("/exp/sbnd/data/users/castalyf/Omnifold_data/t2k-nd280-numu-cc0pi-training-sample/NuMuCC0pi_true_v2.root")
recoWeightsFile = uproot.open("/exp/sbnd/data/users/castalyf/Omnifold_data/t2k-nd280-numu-cc0pi-training-sample/NuMuCC0pi_reco_v2_weights.root")
truthWeightsFile = uproot.open("/exp/sbnd/data/users/castalyf/Omnifold_data/t2k-nd280-numu-cc0pi-training-sample/NuMuCC0pi_true_v2_weights.root")

reco = dict()
for val in recoFile['selectedEvents'].keys():
    reco[val] = recoFile['selectedEvents'][val].array(library='np')
truth = dict()
for val in truthFile['selectedEvents'].keys():
    truth[val] = truthFile['selectedEvents'][val].array(library='np')
recoSystWeightsAll = recoWeightsFile['syst_weights']['weight_arr'].array(library='np')
truthSystWeights = truthWeightsFile['syst_weights']['weight_arr'].array(library='np')

mc_mask = reco['reaction'] < 5
mc_truth_mask = truth['reaction'] < 5

pmu_true = np.sqrt(truth['pmutruevec'][:,1]**2 + truth['pmutruevec'][:,2]**2 + truth['pmutruevec'][:,3]**2)
cosmu_true = truth['pmutruevec'][:,3] / pmu_true
phimu_true = np.arctan2(truth['pmutruevec'][:,2],truth['pmutruevec'][:,1])
pp_true = np.sqrt(truth['pptruevec'][:,1]**2 + truth['pptruevec'][:,2]**2 + truth['pptruevec'][:,3]**2)
cospr_true = truth['pptruevec'][:,3] / pp_true
phip_true = np.arctan2(truth['pptruevec'][:,2],truth['pptruevec'][:,1])
pt_imbalance_true = np.sqrt((truth['pmutruevec'][:,1] + truth['pptruevec'][:,1])**2 + (truth['pmutruevec'][:,2] + truth['pptruevec'][:,2])**2)
mu_p_alpha_true = (-1*truth['pmutruevec'][:,1]*(truth['pmutruevec'][:,1]+truth['pptruevec'][:,1]) - truth['pmutruevec'][:,2]*(truth['pmutruevec'][:,2]+truth['pptruevec'][:,2])) / (np.sqrt(truth['pmutruevec'][:,1]**2 + truth['pmutruevec'][:,2]**2) * pt_imbalance_true)
mu_p_angle_true = (-1*truth['pmutruevec'][:,1]*truth['pptruevec'][:,1] - truth['pmutruevec'][:,2]*truth['pptruevec'][:,2]) / (np.sqrt(truth['pmutruevec'][:,1]**2 + truth['pmutruevec'][:,2]**2) * np.sqrt(truth['pptruevec'][:,1]**2 + truth['pptruevec'][:,2]**2))

all_truth = np.array([pmu_true, cosmu_true, phimu_true, pp_true, cospr_true, phip_true, truth['sample'], truth['evt_id'], truth['Enutrue'], pt_imbalance_true, mu_p_alpha_true, mu_p_angle_true, truth['pptruevec'][:,0]])
truth_weights = np.array(truth['weight'])

for idx, p in enumerate(truth['ppr_true']):
    if p < 0:
        all_truth[3,idx] = -1e6
        all_truth[4,idx] = -1e6
        all_truth[5,idx] = -1e6

mc_truth = all_truth[:, mc_truth_mask]

mc_truth_weights = truth_weights[mc_truth_mask]
truthSystWeights = truthSystWeights[mc_truth_mask]

scaler_pz = StandardScaler()
scaler_pz.fit((mc_truth[0,:]*mc_truth[1,:]).reshape(-1,1))
scaler_muon = StandardScaler()
scaler_muon.fit(np.log(mc_truth[0,:]).reshape(-1,1))
scaler_proton = StandardScaler()
scaler_proton.fit(np.log(mc_truth[3,mc_truth[3,:] >= 0]).reshape(-1,1))
scaler_pt = StandardScaler()
scaler_pt.fit(mc_truth[9,mc_truth[3,:]>=450].reshape(-1,1))

normMuonTruth = np.stack((scaler_muon.transform(np.log(mc_truth[0,:]).reshape(-1,1)).reshape(-1), mc_truth[1,:], mc_truth[2,:], scaler_pz.transform((mc_truth[0,:]*mc_truth[1,:]).reshape(-1,1)).reshape(-1)), axis=1)
normProtonTruth = np.stack((scaler_proton.transform(np.log(mc_truth[3,:]).reshape(-1,1)).reshape(-1), mc_truth[4,:], mc_truth[5,:], scaler_pt.transform(mc_truth[9,:].reshape(-1,1)).reshape(-1), mc_truth[10,:], mc_truth[11,:]), axis=1)

topologies = truth['topology'][mc_truth_mask]
for idx in range(normProtonTruth.shape[0]):
    for k in range(normProtonTruth.shape[1]):
        if mc_truth[-1, idx] < 0 or (topologies[idx] != 1 and topologies[idx] != 2):
            normProtonTruth[idx, k] = 0

truth_stack = np.stack((mc_truth[0,:], mc_truth[1,:], mc_truth[2,:], mc_truth[3,:], mc_truth[4,:], mc_truth[5,:], mc_truth[6,:], truth['reaction'][mc_truth_mask], truth['topology'][mc_truth_mask], truth['neutintcode'][mc_truth_mask], mc_truth[9,:], mc_truth[10,:], mc_truth[11,:]), axis=1)

truth_topology_input = np.array([[int(val==sampleID) for val in topologies] for sampleID in range(7)])
mc_truth_sample_inputs = np.array([[int(val==sampleID) for val in mc_truth[6,:]] for sampleID in range(8)])

norm_mc_truth = np.stack((normMuonTruth[:,0], normMuonTruth[:,1], normMuonTruth[:,2], normProtonTruth[:,0], normProtonTruth[:,1], normProtonTruth[:,2], truth_topology_input[0,:], truth_topology_input[1,:], truth_topology_input[2,:], truth_topology_input[3,:], truth_topology_input[4,:], truth_topology_input[5,:], truth_topology_input[6,:], normProtonTruth[:,3], normProtonTruth[:,4], normProtonTruth[:,5]), axis=1)


np.save("%s/mc_vals_truth.npy" % outputDir, norm_mc_truth[:,0:13])
np.save("%s/mc_vals_truth_NoNorm.npy" % outputDir, truth_stack)
np.save("%s/mc_vals_truth_Unifold0.npy" % outputDir, norm_mc_truth[:,np.r_[0:2,6:13]])
np.save("%s/mc_vals_truth_Unifold1.npy" % outputDir, norm_mc_truth[:,np.r_[13:14,6:13]])
np.save("%s/mc_vals_truth_Unifold2.npy" % outputDir, norm_mc_truth[:,np.r_[14:15,6:13]])
np.save("%s/mc_vals_truth_Unifold3.npy" % outputDir, norm_mc_truth[:,np.r_[15:16,6:13]])
np.save("%s/mc_vals_truth_Multifold.npy" % outputDir, norm_mc_truth[:,np.r_[0:2,3:4,13:16,6:13]])

for throw in range(1):
    pmu_reco = np.sqrt(reco['pmurecovec'][:,1]**2 + reco['pmurecovec'][:,2]**2 + reco['pmurecovec'][:,3]**2)
    cosmu_reco = reco['pmurecovec'][:,3] / pmu_reco
    phimu_reco = np.arctan2(reco['pmurecovec'][:,2],reco['pmurecovec'][:,1])
    pp_reco = np.sqrt(reco['pprecovec'][:,1]**2 + reco['pprecovec'][:,2]**2 + reco['pprecovec'][:,3]**2)
    cospr_reco = reco['pprecovec'][:,3] / pp_reco
    phip_reco = np.arctan2(reco['pprecovec'][:,2],reco['pprecovec'][:,1])
    pt_imbalance_reco = np.sqrt((reco['pmurecovec'][:,1] + reco['pprecovec'][:,1])**2 + (reco['pmurecovec'][:,2] + reco['pprecovec'][:,2])**2)
    mu_p_theta_reco = (-1*reco['pmurecovec'][:,1]*(reco['pmurecovec'][:,1]+reco['pprecovec'][:,1]) - reco['pmurecovec'][:,2]*(reco['pmurecovec'][:,2]+reco['pprecovec'][:,2])) / (np.sqrt(reco['pmurecovec'][:,1]**2 + reco['pmurecovec'][:,2]**2) * pt_imbalance_reco)
    mu_p_angle_reco = (-1*reco['pmurecovec'][:,1]*reco['pprecovec'][:,1] - reco['pmurecovec'][:,2]*reco['pprecovec'][:,2]) / (np.sqrt(reco['pmurecovec'][:,1]**2 + reco['pmurecovec'][:,2]**2) * np.sqrt(reco['pprecovec'][:,1]**2 + reco['pprecovec'][:,2]**2))
    
    all_reco = np.array([pmu_reco, cosmu_reco, phimu_reco, reco['ppr_reco'][:], cospr_reco, phip_reco,reco['sample'][:], reco['evt_id'], reco['Enutrue'], reco['reaction'], reco['topology'], pt_imbalance_reco, mu_p_theta_reco, mu_p_angle_reco, reco['pmu_reco'], pp_reco])
    all_weights = np.array(reco['weight'])
    for idx, sample in enumerate(all_reco[6,:]):
        if sample >= 6:
            all_reco[3,idx] = -1e6
            all_reco[4,idx] = -1e6
            all_reco[5,idx] = -1e6
            all_reco[11,idx] = -1e6
            all_reco[12,idx] = -1e6
            all_reco[13,idx] = -1e6
        if sample == 9:
            all_reco[6,idx] = 2            
    for idx, p in enumerate(all_reco[3,:]):
        if p < 0:
            all_reco[3,idx] = -1e6
            all_reco[4,idx] = -1e6
            all_reco[5,idx] = -1e6
            all_reco[11,idx] = -1e6
            all_reco[12,idx] = -1e6
            all_reco[13,idx] = -1e6

    mc_reco = all_reco[:, mc_mask]
    mc_reco_weights = all_weights[mc_mask]
    recoSystWeights = recoSystWeightsAll[mc_mask,:]

    print(mc_reco.shape)
    mask = mc_reco[-2,:]>2
    mask = np.logical_and(mask, mc_reco[-2,:]<3e4)
    mask = np.logical_and(mask, mc_reco[3,:]<3e4)
    mask = np.logical_and(mask, mc_reco[6,:] > -1)
    mask = np.logical_and(mask, mc_reco[6,:] < 8)
    mc_reco = mc_reco[:, mask]
    mc_reco_weights = mc_reco_weights[mask]
    recoSystWeights = recoSystWeights[mask]
    print(mc_reco.shape)
    
    mask = [a[0] or a[1] for a in zip(abs(mc_reco[-1,:] - mc_reco[3,:]) < 1, mc_reco[3,:]<0)]
    mc_reco = mc_reco[:, mask]
    mc_reco_weights = mc_reco_weights[mask]
    recoSystWeights = recoSystWeights[mask]	
    print(mc_reco.shape)
    
    for idx, val in enumerate(mc_reco_weights):
        if val > 10:
            mc_reco_weights[idx] = 10
        elif val < 0:
            mc_reco_weights[idx] = 0
	        
                
    ids = dict()
    duplicate=0
    for val in list(zip(mc_reco[7,:], mc_reco[8,:])):
        if val[0] in ids.keys():
            duplicate = duplicate+1
            ids[val[0]].append(val[1])
        else:
            ids[val[0]] =  [val[1]]
            	            
    mc_pass_truth = [False for val in mc_truth[0,:]]
    for idx in range(mc_truth.shape[1]):
        thisID = mc_truth[7, idx]
        if thisID in ids.keys():
            if len(ids[thisID]) > 1:
                print("Repeat ID %d" % thisID)
                continue
            for nuIdx in range(len(ids[thisID])):
                if abs(mc_truth[8, idx] - ids[thisID][nuIdx]) < 2:
                    del ids[thisID][nuIdx]
                    mc_pass_truth[idx] = True
                    break
                else:
                    print("ID %d at truth idx %d has mismatched Enutrue %.1f %.1f" % (thisID, idx, mc_truth[8,idx], ids[thisID][nuIdx]))
    mc_pass_reco = np.array([True for val in mc_reco[0,:]])
    for idx, val in enumerate(list(zip(mc_reco[7,:], mc_reco[8,:]))):
        for enu in ids[val[0]]:
            if abs(val[1] - enu) < 2:
                mc_pass_reco[idx] = False

    
    matchReco = mc_reco[:,mc_pass_reco]
    matchTruth = mc_truth[:,mc_pass_truth]
    for idx, val in enumerate(list(zip(matchReco[8, :], matchTruth[8,:]))):
        if abs(val[0] - val[1]) > 2:
            if abs(val[1] - matchReco[8,idx+1]) < 2:
                tmpArr = [val for val in matchReco[:,idx+1]]
                for r in range(matchReco.shape[0]):
                    matchReco[r,idx+1] = matchReco[r,idx]
                    matchReco[r,idx] = tmpArr[r]
                                                                         
    normMuonMC = np.stack((scaler_muon.transform(np.log(matchReco[0,:]).reshape(-1,1)).reshape(-1), matchReco[1,:], matchReco[2,:], scaler_pz.transform((matchReco[0,:]*matchReco[1,:]).reshape(-1,1)).reshape(-1)), axis=1)
    normProtonMC = np.stack((scaler_proton.transform(np.log(matchReco[3,:]).reshape(-1,1)).reshape(-1), matchReco[4,:], matchReco[5,:], scaler_pt.transform(matchReco[11,:].reshape(-1,1)).reshape(-1), matchReco[12,:], matchReco[13,:]), axis=1)
    for idx in range(normProtonMC.shape[0]):
        for k in range(normProtonMC.shape[1]):
            if matchReco[3, idx] < 0:
                normProtonMC[idx, k] = 0

    mc_reco_sample_input = np.array([[int(val==sampleID) for val in matchReco[6,:]] for sampleID in range(8)])
    norm_mc_reco = np.stack((normMuonMC[:,0], normMuonMC[:,1], normMuonMC[:,2], normProtonMC[:,0], normProtonMC[:,1], normProtonMC[:,2], mc_reco_sample_input[0,:],mc_reco_sample_input[1,:], mc_reco_sample_input[2,:], mc_reco_sample_input[3,:], mc_reco_sample_input[4,:], mc_reco_sample_input[5,:], mc_reco_sample_input[6,:], mc_reco_sample_input[7,:], normProtonMC[:,3], normProtonMC[:,4], normProtonMC[:,5]), axis=1)


    np.save("%s/mc_pass_truth.npy" % (outputDir), mc_pass_truth)
    np.save("%s/mc_vals_reco.npy" % (outputDir), norm_mc_reco[:,0:14])
    np.save("%s/mc_vals_reco_Unifold0.npy" % (outputDir), np.stack((normMuonMC[:,0], normMuonMC[:,1], mc_reco_sample_input[0,:],mc_reco_sample_input[1,:], mc_reco_sample_input[2,:], mc_reco_sample_input[3,:], mc_reco_sample_input[4,:], mc_reco_sample_input[5,:], mc_reco_sample_input[6,:], mc_reco_sample_input[7,:]), axis=1))
    np.save("%s/mc_vals_reco_Unifold1.npy" % (outputDir), np.stack((normProtonMC[:,3], mc_reco_sample_input[0,:],mc_reco_sample_input[1,:], mc_reco_sample_input[2,:], mc_reco_sample_input[3,:], mc_reco_sample_input[4,:], mc_reco_sample_input[5,:], mc_reco_sample_input[6,:], mc_reco_sample_input[7,:]), axis=1))
    np.save("%s/mc_vals_reco_Unifold2.npy" % (outputDir), np.stack((normProtonMC[:,4], mc_reco_sample_input[0,:],mc_reco_sample_input[1,:], mc_reco_sample_input[2,:], mc_reco_sample_input[3,:], mc_reco_sample_input[4,:], mc_reco_sample_input[5,:], mc_reco_sample_input[6,:], mc_reco_sample_input[7,:]), axis=1))
    np.save("%s/mc_vals_reco_Unifold3.npy" % (outputDir), np.stack((normProtonMC[:,5], mc_reco_sample_input[0,:],mc_reco_sample_input[1,:], mc_reco_sample_input[2,:], mc_reco_sample_input[3,:], mc_reco_sample_input[4,:], mc_reco_sample_input[5,:], mc_reco_sample_input[6,:], mc_reco_sample_input[7,:]), axis=1))
    np.save("%s/mc_vals_reco_Multifold.npy" % (outputDir), norm_mc_reco[:,np.r_[0:2,3:4,14:17,6:14]])
    np.save("%s/mc_weights_reco.npy" % (outputDir), mc_reco_weights[mc_pass_reco])
    np.save("%s/mc_weights_truth.npy" % (outputDir), mc_truth_weights)
    np.save("%s/mc_pass_reco.npy" % (outputDir), mc_pass_reco[mc_pass_reco])
    
    for i in range(100):
        np.save("%s/mc_weights_reco_Throw%d.npy" % (outputDir, i), mc_reco_weights[mc_pass_reco] * recoSystWeights[mc_pass_reco, i])
        np.save("%s/mc_weights_truth_Throw%d.npy" % (outputDir, i), mc_truth_weights  * truthSystWeights[:, i])

