import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import social_stgcnn_with_transformer
from sentence_transformers import SentenceTransformer
import copy
import csv

# Step 1: Load the pre-trained Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Define the text_to_embedding function using the SBERT model
def sbert_text_to_embedding(text):
    return sbert_model.encode(text, convert_to_tensor=True)  # Convert text to tensor embeddings

def test(KSTEPS=100):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0 
    for batch in loader_test: 
        step+=1
        #Get data
        batch = [tensor.to("cpu") for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr, text_features = batch


        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze(), text_features)
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0,2,3,1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]

        #For now I have my bi-variate parameters 
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).to("cpu")
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]

        # mean, cov 저장
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []
        raw_data_dict[step]['mean'] = []
        raw_data_dict[step]['cov'] = []
        raw_data_dict[step]['abs_positions_mean'] = [] 

        #mean cov
        abs_positions = nodes_rel_to_nodes_abs(mean.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())
        raw_data_dict[step]['mean'].append(mean.data.cpu().numpy())
        raw_data_dict[step]['cov'].append(cov.data.cpu().numpy())
        raw_data_dict[step]['abs_positions_mean'].append(abs_positions)


            
        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()



            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)

    
    save_full_data_to_csv(raw_data_dict, filename="mean_cov_abs.csv")
    save_raw_data_dict_to_csv(raw_data_dict, "raw_data_dict.csv")
    return ade_,fde_,raw_data_dict

def save_full_data_to_csv(raw_data_dict, filename="mean_cov_abs_data.csv"):
    """
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # CSV 헤더 작성
        writer.writerow([
            "Step", "Sequence", "Object", 
            "meanX", "meanY", 
            "mean_abs_X", "mean_abs_Y", 
            "C00", "C01", "C10", "C11"
        ])
        
        for step, data in raw_data_dict.items():
            if 'mean' in data and 'abs_positions_mean' in data and 'cov' in data:
                for seq_num, (mean_seq, abs_mean_seq, cov_seq) in enumerate(
                    zip(data['mean'], data['abs_positions_mean'], data['cov'])
                ):
                    num_objects = min(mean_seq.shape[1], abs_mean_seq.shape[1], cov_seq.shape[1])
                    num_sequences = mean_seq.shape[0]

                    for obj_idx in range(num_objects): 
                        for seq_idx in range(num_sequences):  
                      
                            mean_coords = mean_seq[seq_idx, obj_idx]
                            mean_coords = mean_coords if mean_coords.shape == (2,) else [0.0, 0.0]

                   
                            abs_coords = abs_mean_seq[seq_idx, obj_idx]
                            abs_coords = abs_coords if isinstance(abs_coords, (np.ndarray, list)) and len(abs_coords) == 2 else [0.0, 0.0]

                       
                            cov_matrix = cov_seq[seq_idx, obj_idx]
                            cov_matrix = cov_matrix if cov_matrix.shape == (2, 2) else np.zeros((2, 2))

                            writer.writerow([
                                step, seq_idx, obj_idx,
                                mean_coords[0], mean_coords[1],  # meanX, meanY
                                abs_coords[0], abs_coords[1],   # mean_abs_X, mean_abs_Y
                                cov_matrix[0, 0], cov_matrix[0, 1],  # C00, C01
                                cov_matrix[1, 0], cov_matrix[1, 1]   # C10, C11
                            ])
    
    print(f"Mean, absolute mean, and covariance data saved to {filename}")

def save_raw_data_dict_to_csv(raw_data_dict, filename="raw_data_dict.csv"):
    """
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Step", "Type", "Sequence", "Object", "X", "Y"])
        
        for step, data in raw_data_dict.items():

            for seq_idx, seq in enumerate(data['obs']):
                for obj_idx, coords in enumerate(seq):
                    writer.writerow([step, "obs", seq_idx, obj_idx, coords[0], coords[1]])


            for seq_idx, seq in enumerate(data['trgt']):
                for obj_idx, coords in enumerate(seq):
                    writer.writerow([step, "trgt", seq_idx, obj_idx, coords[0], coords[1]])


            for sample_idx, sample in enumerate(data['pred']):  
                for seq_idx, seq in enumerate(sample):         
                    for obj_idx, coords in enumerate(seq):      
                        writer.writerow([step, f"pred_{sample_idx}", seq_idx, obj_idx, coords[0], coords[1]])
    
    print(f"raw_data_dict saved to {filename}")

paths = ['./checkpoint/*social-stgcnn*']
KSTEPS=100

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)




for feta in range(len(paths)):
    ade_ls = [] 
    fde_ls = [] 
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f: 
            cm = pickle.load(f)
        print("Stats:",cm)



        #Data prep     
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = './datasets/'+args.dataset+'/'

        dset_test = TrajectoryDatasetWithText(
                data_dir=data_set + 'test/test.txt',  # Path to trajectory data
                text_file=data_set + 'test/test_text.txt',  # Path to text data
                obs_len=obs_seq_len,                    # Observation length
                pred_len=pred_seq_len,                  # Prediction length
                skip=36,                                # Skip frames
                norm_lap_matr=True,                     # Normalize Laplacian matrix
                text_to_embedding=sbert_text_to_embedding)  # Use SBERT for text embeddings
        
        loader_test = DataLoader(
                dset_test,
                batch_size=1,#This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=0)



        #Defining the model 
        model = social_stgcnn_with_transformer(
                n_stgcnn=args.n_stgcnn,
                n_txpcnn=args.n_txpcnn,
                input_feat=args.input_size,
                output_feat=args.output_size,
                seq_len=args.obs_seq_len,
                pred_seq_len=args.pred_seq_len,
                kernel_size=args.kernel_size
#    text_feat_dim=args.text_feat_dim,
#    text_embedding_dim=args.text_embedding_dim
                ).to("cpu")
        # Load state_dict with missing keys
        state_dict = torch.load(model_path)
        keys_to_remove = ["feature_fusion.weight", "feature_fusion.bias"]
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]

        # Load the modified state_dict
        model.load_state_dict(state_dict, strict=False)



        ade_ =999999
        fde_ =999999
        print("Testing ....")
        ad,fd,raw_data_dic_= test()
        ade_= min(ade_,ad)
        fde_ =min(fde_,fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        print("ADE:",ade_," FDE:",fde_)




    print("*"*50)

    print("Avg ADE:",sum(ade_ls)/5)
    print("Avg FDE:",sum(fde_ls)/5)
