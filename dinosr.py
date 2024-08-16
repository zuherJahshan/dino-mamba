
import torch
import torch.nn as nn
from fairseq.modules import LayerNorm
import copy
from utils import OutConvShape
from codebook import Codebook
import torch.nn.functional as F
from einops import einsum, rearrange
from utils import get_starting_mask_prob
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)
from typing import Callable
import multiprocessing as mp
from mamba import DeepMamba

mp.set_start_method('spawn', force=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DinoSR(nn.Module):
    def __init__(
        self,
        cfg: Wav2Vec2Config,
        model_creator: Callable,
    ):
        super(DinoSR, self).__init__()

        # save the config
        self.cfg = cfg
        self.teacher_decay = cfg.ema_decay
        self.teacher_end_decay = cfg.ema_end_decay
        self.teacher_decay_steps = cfg.ema_anneal_end_step
        self.kl_div_from_uniform_lambda = cfg.kl_div_from_uniform_lambda
        self.starting_mask_prob = get_starting_mask_prob(cfg.mask_prob, cfg.mask_length)
        self.num_layers = cfg.encoder_layers
        self.layers_to_include_in_loss = cfg.average_top_k_layers
        self.mask_length = cfg.mask_length

        # Initailize the feature extractor
        conv_feature_extractor_layers = eval(cfg.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel( 
            conv_layers=conv_feature_extractor_layers,
            dropout=0.0,
            mode=cfg.extractor_mode # Check this
        ).to(device) # layer included
        
        conv_feature_layers_list = eval(cfg.conv_feature_layers) # layer included

        self.conv_feature_layers = [
            {
                'd': layer[0],
                'k': layer[1],
                's': layer[2],
            } for layer in conv_feature_layers_list
        ]


        self.feature_extractor_out_length_calculator = OutConvShape(
            self.conv_feature_layers,
            same=False,
        ) # no nn.module, but nothing to be saved here

        # TODO: understand what's happening here
        normalized_shape = conv_feature_extractor_layers[-1][0]
        self.layer_norm = LayerNorm(
            normalized_shape
        )# layer included

        self.post_fe_proj = nn.Linear(
            in_features=self.conv_feature_layers[-1]['d'],
            out_features=cfg.encoder_embed_dim,
        )# layer included

        # apply dropout
        self.dropout = nn.Dropout(cfg.dropout)# layer included

        # layer included
        self.student = model_creator(cfg)
        # layers included
        self.classifiers = nn.ModuleList([
            nn.Linear(cfg.encoder_embed_dim, cfg.codebook_size) for _ in range(cfg.average_top_k_layers)
        ])

        # self.softmax = nn.Softmax(dim=-1) # Can be discarded if returning the log softmax function
        
        # layer included
        self.teacher = model_creator(cfg)
        
        # layer included
        self.teacher.load_state_dict(self.student.state_dict())
        self.codebook = Codebook(
            dim=cfg.encoder_embed_dim,
            num_codewords=cfg.codebook_size,
            layers=cfg.average_top_k_layers,
        )
    
    
    def _mask(self, features, feature_lengths):
        # 1. flip a coin (with mask_prob) for each element in every feature determining if it would be a beginning of a masked sequence.
        # the shape of the flipped coin will be [B,T]
        starting_masks = (torch.rand(features.shape[0], features.shape[1]) < self.starting_mask_prob).to(device).to(torch.int8)
        
        # the ending_masks should preserve the following property:
        # ending_mask[i, j] = starting_mask[i, j-mask_length], and if j-mask_length is out of bounds then 0
        ending_masks = torch.zeros_like(starting_masks).to(device).to(torch.int8)
        ending_masks[:, self.mask_length:] = starting_masks[:, :-self.mask_length]

        # 2. cover the next bases_to_cover bases from each point
        # bases to cover is the cumulative sum of the tensor starting_masks - ending_masks
        bases_to_cover = torch.cumsum(starting_masks - ending_masks, dim=1) > 0

        # check that there is no minus values in starting_masks - ending_masks
        assert (bases_to_cover).min() >= 0

        # 3. zero every mask that falls out of the feature_lengths bounds
        included_indices = torch.arange(features.shape[1]).to(device).unsqueeze(0).expand(features.shape[0], -1) <= feature_lengths.unsqueeze(1)
        
        return included_indices & bases_to_cover


    # gets a waveform (B,T) and outputs a compressed representation (B,T',D)
    def _before_encoder(
        self,
        waveforms
    ):
        # 1. run the feature extractor and transpose features
        features = self.feature_extractor(waveforms)
        
        # 2. run layer norm
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        # 4. run post fature extractor proj
        features = self.post_fe_proj(features)
        return features


    def forward(
        self,
        waveforms,
        lengths,
        only_student=False
    ):
        result = {}

        # 1. Apply the feature extractor with adaptations to the input
        features = self._before_encoder(waveforms)

        # 2. run conv out shape to get the true output true lengths
        lengths = self.feature_extractor_out_length_calculator(lengths)
        result["out_lengths"] = lengths

        # 3. mask features (up till lengths)
        mask = self._mask(features, lengths) # will get back a tensor B,T
        flattened_mask = rearrange(mask, "b t -> (b t)") # flattens the tensor to be [B*T]

        # 4. clone features to teacher before applying mask
        cloned_features = features.clone()

        # 5. Apply mask and run the student model
        masked_features = einsum(features, mask, "b t d, b t -> b t d")
        x, student_layer_results = self.student(masked_features) # [B,T/C,D]

        # check that the model returns a tensor of the shape [T,B,D]
        assert x.shape[0] == features.shape[1] and x.shape[1] == features.shape[0] and x.shape[2] == features.shape[2],\
            f"Model returned a tensor of shape {x.shape} while expected [{features.shape[1]},{features.shape[0]},{features.shape[2]}]"
        
        x = rearrange(x, "t b d -> b t d")
        
        student_layer_results = [rearrange(layer_result[2], "t b d -> b t d") for layer_result in student_layer_results]
        
        # flatten and apply mask
        flattened_x = rearrange(x, "b t d -> (b t) d")[flattened_mask]
        flattened_student_layer_results = [rearrange(layer_result, "b t d -> (b t) d")[flattened_mask] for layer_result in student_layer_results]
        result["student"] = {
            "x": x,
            "layer_results": student_layer_results,
        }

        if only_student:
            return result

        # 8. run the teacher model without training - make sure the pre teacher is not trainable here
        with torch.no_grad():
            self.teacher.eval()
            y, teacher_layer_results = self.teacher(cloned_features)

            y = F.instance_norm(rearrange(y, "t b d -> b d t"))
            y = rearrange(y, "b d t -> b t d")
            
            # rearange to BDT perform instance norm then rearrange back to BTD
            teacher_layer_results = [rearrange(layer_result[2], "t b d -> b d t") for layer_result in teacher_layer_results]
            teacher_layer_results = [F.instance_norm(layer_result.float()) for layer_result in teacher_layer_results]
            teacher_layer_results = [rearrange(layer_result, "b d t -> b t d") for layer_result in teacher_layer_results]

            # flatten and apply mask
            flattened_y = rearrange(y, "b t d -> (b t) d")[flattened_mask]
            flattened_teacher_layer_results = [rearrange(layer_result, "b t d -> (b t) d")[flattened_mask] for layer_result in teacher_layer_results]
            result["teacher"] = {
                "x": y,
                "layer_results": teacher_layer_results,
            }

        # 9. get the closest codewords, and update the codeword
        targets = [None] * self.layers_to_include_in_loss
        first_layer_to_include = self.num_layers - self.layers_to_include_in_loss
        result["codebook_update"] = {}
        for i in range(first_layer_to_include, self.num_layers):
            # get targets
            closest_codewords = self.codebook.get_closest_codewords(
                flattened_teacher_layer_results[i],
                i - first_layer_to_include
            )
            # return information re codebook updates
            result["codebook_update"].update({
                i - first_layer_to_include: {
                    "closest_codewords": closest_codewords,
                    "flattened_teacher_layer_results": flattened_teacher_layer_results[i],
                }
            })
            targets[i - first_layer_to_include] = closest_codewords

        
        def calculate_accuracy(representation, target):
        #     # pred is [B,C], target is [B]
        #     #we should include only 
            return ((representation.argmax(dim=-1) == target).float()).sum() / target.shape[0]
        
        def calculate_probability_bins(representation, binary=False):
            # rep is [B,C]
            # onehot the target to have the same shape as pred
            representation = F.one_hot(representation.argmax(dim=-1), num_classes=representation.shape[-1]).to(torch.float32)
            histogram = representation.sum(dim=0)
            
            if binary:
                return (histogram > 0)

            return histogram / representation.shape[0]

        def calculate_loss(representation, target):
            # representation is [B,C], target is [B]
            
            # onehot the target to have the same shape as pred
            pred = -1*F.log_softmax(representation, dim=-1)
            onehot_target = F.one_hot(target, num_classes=pred.shape[-1]).to(torch.float32)
            # calculate the loss
            return einsum(onehot_target, pred, "b c, b c ->")
        
        def calculate_kl_divergence_regularization(representation):
            # Calculate the empirical distribution of cluster assignments
            cluster_assignments = F.softmax(representation, dim=-1) # B,C
            # apply mask
            # cluster_assignments = einsum(cluster_assignments, mask, "b t c, b t -> b t c")
            
            cluster_probs = torch.sum(cluster_assignments, dim=0) / cluster_assignments.shape[0]
            
            # assert all values of cluster_probs are between 0 and 1
            assert (cluster_probs >= 0).all() and (cluster_probs <= 1).all()
            
            # Calculate the uniform distribution
            uniform_distribution = torch.ones_like(cluster_probs) / cluster_assignments.shape[1]
            
            # Calculate KL Divergence
            kl_div = F.kl_div(cluster_probs.log(), uniform_distribution, reduction='sum')
            
            return kl_div
        

        # def calculate_over_clustering_penalty(representation, mask):
        #     # Softmax over the classes to get soft assignments
        #     soft_assignments = F.softmax(representation, dim=-1)
            
        #     # Sum soft assignments across batch and time dimensions
        #     cluster_sums = einsum(soft_assignments, mask.float(), "b t c, b t -> c")
            
        #     # Calculate the mean and variance of the cluster assignments
        #     mean_cluster_assignments = cluster_sums.mean()
        #     var_cluster_assignments = cluster_sums.var()
            
        #     # Penalty for deviation from the mean (over-clustering penalty)
        #     over_clustering_penalty = var_cluster_assignments / (mean_cluster_assignments ** 2 + 1e-6)
            
        #     return over_clustering_penalty
            

        # 11. calculate the loss
        loss = 0
        accuracy = 0
        kl_divergence_regularization = 0
        for i in range(self.layers_to_include_in_loss):
            representations = self.classifiers[i](flattened_student_layer_results[i])
            loss += calculate_loss(representations, targets[i])
            accuracy += calculate_accuracy(representations, targets[i])
             
            # Calculate the KL Divergence regularization
            kl_divergence_regularization += calculate_kl_divergence_regularization(representations)

        result['prob_bins'] = calculate_probability_bins(representations)
        result['prob_bins_binary'] = calculate_probability_bins(representations, binary=True)

        ce_loss = loss / self.layers_to_include_in_loss
        kl_loss = self.kl_div_from_uniform * kl_divergence_regularization / self.layers_to_include_in_loss
        loss = ce_loss + kl_loss
        accuracy = accuracy / self.layers_to_include_in_loss
        result["loss"] = loss
        result["cross_entropy_loss"] = ce_loss
        result["kl_divergence_loss"] = kl_loss
        result["accuracy"] = accuracy
        result["targets"] = targets[-1]

        return result


    def update_teacher_params(
        self,
        batch_step    
    ):
        # gamma will linearly scale up from self.teacher_decay to self.teacher_end_decay for 30K steps
        gamma = self.teacher_decay +\
            min((batch_step / self.teacher_decay_steps), 1) * (self.teacher_end_decay - self.teacher_decay)
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data = teacher_param.data * gamma + student_param.data * (1 - gamma)


    def save(self, filepath):
        state = {
            'model_state_dict': self.state_dict(),
            'codebook_state': self.codebook.save_state(),
        }
        torch.save(state, filepath)


    def load(self, filepath):
        state = torch.load(filepath)
        self.load_state_dict(state['model_state_dict'])
        self.codebook.load_state(state['codebook_state'])

def model_creator(cfg):
    if cfg.model_type == "transformer":
        return TransformerEncoder(cfg)
    else:
        return DeepMamba(cfg)