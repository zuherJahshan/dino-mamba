from dataclasses import dataclass, field
from fairseq.models.wav2vec import Wav2Vec2Config
from typing import Optional
from omegaconf import II

@dataclass
class DinosrAudioConfig(Wav2Vec2Config):

    encoder_layers: int = field(default=4)
    extractor_mode: str = field(default='layer_norm')
    encoder_embed_dim: int = field(default=256)
    encoder_ffn_embed_dim: int = field(default=512)
    encoder_attention_heads: int = field(default=8)

    discrete: bool = field(default=False)
    codebook_size: int = field(default=256)
    normal_init_codebook: bool = field(default=False)
    codebook_init_decay: float = field(default=0.9)
    codebook_end_decay: float = field(default=0.9)
    codebook_end_decay_step: int = field(default=0)
    freeze_teacher_step: int = field(
        default=200001, metadata={"help": "step to freeze teacher"}
    )
    freeze_pre_enc_modules: bool = field(
        default=True, metadata={"help": "when freezing teacher, freeze the CNN extractor as well"}
    )
    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=4, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = 30_000

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )