import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_array(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

# 텍스트 파일을 읽는 함수 (두 번째 열만 추출)
def read_text_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return [line.strip().split('\t')[1] for line in lines]

class TrajectoryDatasetWithText(Dataset):
    """Dataloader for the Trajectory datasets with text support"""
    def __init__(
        self, data_dir, text_file, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t', norm_lap_matr=True, text_to_embedding=None):
        """
        Args:
        - data_dir: Directory containing trajectory dataset files
        - text_file: File containing two columns: <ID> and <Text>
        - text_to_embedding: Function to convert text to embeddings
        """
        super(TrajectoryDatasetWithText, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.text_file = text_file
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.text_to_embedding = text_to_embedding
        
        #all_files = os.listdir(self.data_dir)
        #all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        # Prepare Sequences
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        # Trajectory Data Loading
        data = read_file(self.data_dir, delim)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = [data[frame == data[:, 0], :] for frame in frames]
        num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

        # Text Data Loading
        self.text_data = read_text_file(self.text_file)
        assert len(self.text_data) >= num_sequences, "Mismatch between text and trajectory sequences"
 
        for idx in range(0, num_sequences * self.skip + 1, skip):
            if idx + self.seq_len > len(frame_data):  # 초과 여부 확인
                print(f"Skipping sequence at idx {idx}: out of bounds")
                continue
            curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))

            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
            num_peds_considered = 0
            _non_linear_ped = []

            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                if pad_end - pad_front != self.seq_len:
                    continue
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                _idx = num_peds_considered
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                num_peds_considered += 1

            if num_peds_considered > min_ped:
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_list.append(curr_seq[:num_peds_considered])
                seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                non_linear_ped += _non_linear_ped

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert to Torch tensors
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Process Graphs and Text
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        self.text_features = []

        print("Processing Data ...")
        pbar = tqdm(total=len(self.seq_start_end), desc="Processing Sequences")
        for i, (start, end) in enumerate(self.seq_start_end):
            if start < 0 or end > self.obs_traj.shape[0]:
                print(f"Skipping sequence at idx {i}: out of bounds (start: {start}, end: {end})")
                pbar.update(1)
                continue

            if i >= len(self.text_data):
                print(f"Skipping sequence at idx {i}: text data out of range")
                pbar.update(1)
                continue

            print(f"Processing sequence at idx {i}: start={start}, end={end}")
            v_obs, a_obs = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            v_pred, a_pred = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_obs.append(v_obs.clone())
            self.A_obs.append(a_obs.clone())
            self.v_pred.append(v_pred.clone())
            self.A_pred.append(a_pred.clone())

            text_embedding = self.text_to_embedding(self.text_data[i])
            print(f"text_embedding shape at idx {i}: {text_embedding.shape}")
            self.text_features.append(text_embedding)

            pbar.update(1)

        self.text_features = torch.stack(self.text_features)
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.text_features[index]]