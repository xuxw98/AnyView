import torch
from torch import Tensor, nn
import torch.nn.functional as F

from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import gather_operation
from models.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT,
                            get_clones)


class VoteNetDecoder(nn.Module):
    def __init__(self, vote_factor=1, decoder_feature_dim=256, num_proposal=256):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = decoder_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.num_proposal = num_proposal
        self.conv1 = torch.nn.Conv1d(self.in_dim, decoder_feature_dim, 1)
        self.conv2 = torch.nn.Conv1d(decoder_feature_dim, decoder_feature_dim, 1)
        self.conv3 = torch.nn.Conv1d(decoder_feature_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(decoder_feature_dim)
        self.bn2 = torch.nn.BatchNorm1d(decoder_feature_dim)

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[decoder_feature_dim, 128, 128, decoder_feature_dim],
            use_xyz=True,
            normalize_xyz=True
        )
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, (3+out_dim)*vote_factor, num_seed)
                
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim)
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        
        residual_features = net[:,:,:,3:] # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2,1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()

        features_norm = torch.norm(vote_features, p=2, dim=1)
        vote_features = vote_features.div(features_norm.unsqueeze(1))

        xyz, features, fps_inds = self.vote_aggregation(vote_xyz, vote_features)
        
        return xyz, features