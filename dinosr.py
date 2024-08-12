
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


    def forward(
        self,
        waveforms,
        lengths,
        only_student=False
    ):
        result = {}
        # 1. run the feature extractor and transpose features
        features = self.feature_extractor(waveforms)
        
        # 2. run layer norm
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        # 3. run conv out shape to get the true output true lengths
        lengths = self.feature_extractor_out_length_calculator(lengths)
        result["out_lengths"] = lengths
        
        # 4. run post fature extractor proj
        features = self.post_fe_proj(features)

        # 5. apply dropout

        # 6. mask features (up till lengths)
        mask = self._mask(features, lengths)

        # 7. run the student model and out the output in the results dict
        # clone features to teacher before applying mask
        cloned_features = features.clone()
        masked_features = mask.unsqueeze(-1) * features
        x, student_layer_results = self.student(masked_features) # [B,T/C,D]
        result["student"] = rearrange(x, "t b d -> b t d")
        result["student_layer_results"] = [rearrange(layer_result[2], "t b d -> b t d") for layer_result in student_layer_results]
        if only_student:
            return result

        # 8. run the teacher model without training - make sure the pre teacher is not trainable here
        with torch.no_grad():
            teacher_x, teacher_layer_results = self.teacher(cloned_features)
            result["teacher"] = rearrange(teacher_x, "t b d -> b t d")
            result["teacher_layer_results"] = [rearrange(layer_result[2], "t b d -> b t d") for layer_result in teacher_layer_results]

        # 9. get the closest codewords, and update the codeword
        targets = []
        first_layer_to_include = self.num_layers - self.layers_to_include_in_loss
        result["codebook_update"] = {}
        for i in range(first_layer_to_include, self.num_layers):
            flattened_teacher_layer_results = F.instance_norm(teacher_layer_results[i][2])
            flattened_teacher_layer_results = rearrange(flattened_teacher_layer_results, "b t d -> (b t) d")
            closest_codewords = self.codebook.get_closest_codewords(
                flattened_teacher_layer_results,
                i - first_layer_to_include
            )
            # return information re codebook updates
            result["codebook_update"].update({
                i - first_layer_to_include: {
                    "closest_codewords": closest_codewords,
                    "flattened_teacher_layer_results": flattened_teacher_layer_results,
                    "flattened_mask": rearrange(mask, "b t -> (b t)"),
                }
            })
            closest_codewords = rearrange(closest_codewords, "(b t) -> b t", b=features.shape[0])
            targets.append(closest_codewords)

        
        def calculate_accuracy(representation, target, mask):
            # pred is [B,T,C], target is [B,T] and mask is [B,T].
            #we should include only 
            return ((representation.argmax(dim=-1) == target).float() * mask.float()).sum() / mask.sum()
        
        def calculate_probability_bins(representation, target, mask, binary=False):
            # pred is [B,T,C], target is [B,T] and mask is [B,T].
            # onehot the target to have the same shape as pred
            onehot_target = F.one_hot(target, num_classes=representation.shape[-1]).to(torch.float32)
            onehot_mask = mask.unsqueeze(-1).expand_as(onehot_target).to(torch.float32)
            # scatter the predicitons into a histogram
            hist_sum = einsum(onehot_mask, onehot_target, F.softmax(representation, dim=-1), "b t c, b t c, b t c -> c")
            if binary:
                return (hist_sum > 0)
            hist_cnt = (onehot_target * onehot_mask).sum(dim=(0,1))
            hist_cnt = torch.where(hist_cnt == 0, 1, hist_cnt)
            return (hist_sum / hist_cnt)

        def calculate_loss(representation, target, mask):
            # pred is [B,T,C], target is [B,T] and mask is [B,T].
            # onehot the target to have the same shape as pred
            pred = -1*F.log_softmax(representation, dim=-1)
            onehot_target = F.one_hot(target, num_classes=pred.shape[-1]).to(torch.float32)
            masked_onehot_target = einsum(onehot_target, mask, "b t c, b t -> b t c")
            # calculate the loss
            return einsum(masked_onehot_target, pred, "b t c, b t c ->")

        def calculate_entropy_regularization(representation, mask):
            # Calculate the softmax to get the probabilities
            probs = F.softmax(representation, dim=-1)
            log_probs = F.log_softmax(representation, dim=-1)
            
            # Calculate entropy: -Σp(x)log(p(x))
            entropy = -einsum(probs, log_probs, "b t c, b t c -> b t") * mask.float()
            
            # Sum over time and batch, then normalize
            return entropy.sum()
        
        def calculate_kl_divergence_regularization(cluster_assignments):
            # Calculate the empirical distribution of cluster assignments
            cluster_probs = torch.mean(cluster_assignments, dim=0)
            
            # Calculate the uniform distribution
            uniform_distribution = torch.ones_like(cluster_probs) / cluster_probs.shape[0]
            
            # Calculate KL Divergence
            kl_div = F.kl_div(cluster_probs.log(), uniform_distribution, reduction='batchmean')
            
            return kl_div
        

        def calculate_over_clustering_penalty(representation, mask):
            # Softmax over the classes to get soft assignments
            soft_assignments = F.softmax(representation, dim=-1)
            
            # Sum soft assignments across batch and time dimensions
            cluster_sums = einsum(soft_assignments, mask.float(), "b t c, b t -> c")
            
            # Calculate the mean and variance of the cluster assignments
            mean_cluster_assignments = cluster_sums.mean()
            var_cluster_assignments = cluster_sums.var()
            
            # Penalty for deviation from the mean (over-clustering penalty)
            over_clustering_penalty = var_cluster_assignments / (mean_cluster_assignments ** 2 + 1e-6)
            
            return over_clustering_penalty
            

        # 11. calculate the loss
        loss = 0
        accuracy = 0
        masks_sum = 0
        # entropy_regularization = 0
        # kl_divergence_regularization = 0
        # reg_lambda = 30
        for i in range(self.layers_to_include_in_loss):
            representation = self.classifiers[i](rearrange(student_layer_results[first_layer_to_include + i][2], "t b c -> b t c")) # [B,T,C] where C is the number of classes
            masks_sum += mask.sum()
            loss += calculate_loss(representation, targets[i], mask)
            accuracy += calculate_accuracy(representation, targets[i], mask)
             
            # Calculate the KL Divergence regularization
            # kl_divergence_regularization += calculate_over_clustering_penalty(representation, mask)

        result['prob_bins'] = calculate_probability_bins(representation, targets[-1], mask)
        result['prob_bins_binary'] = calculate_probability_bins(representation, targets[-1], mask, binary=True)

        loss = 15 * loss / masks_sum
        accuracy = accuracy / self.layers_to_include_in_loss
        result["loss"] = loss
        result["accuracy"] = accuracy
        result["targets"] = targets

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