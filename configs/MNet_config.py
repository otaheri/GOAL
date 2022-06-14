from typing import Tuple, Optional, Any, Union
from loguru import logger
from copy import deepcopy

from dataclasses import dataclass
from omegaconf import OmegaConf

import os, sys
cdir = os.path.dirname(sys.argv[0])


@dataclass
class Metrics:
    v2v: Tuple[str] = ('procrustes',)


@dataclass
class Evaluation:
    body: Metrics = Metrics()

############################## DATASETS ##############################

@dataclass
class Sampler:
    ratio_2d: float = 0.5
    use_equal_sampling: bool = True
    importance_key: str = 'weight'
    balance_genders: bool = True


@dataclass
class NumWorkers:
    train: int = 6
    val: int = 6
    test: int = 0


@dataclass
class Splits:
    train = tuple()
    val = tuple()
    test = tuple()


@dataclass
class DatasetConfig:
    batch_size: int = 32
    use_equal_sampling: bool = True
    use_packed: bool = True
    use_face_contour: bool = True

    splits: Splits = Splits()
    num_workers: NumWorkers = NumWorkers()

    dataset_dir: str = 'GOAL_dataset'
    objects_dir: str = ''
    grab_path: str = ''

    fps: int = 30
    past_frames: int =  10
    future_pred: int = 10
    chunk_size: int = 21

    model_path: str =  ''

    verts_sampled: str = f'{cdir}/../consts/verts_ids_0512.npy'
    verts_feet: str = f'{cdir}/../consts/feet_verts_ids_0512.npy'
    rh2smplx_ids: str = f'{cdir}/../consts/rhand_smplx_ids.npy'
    vertex_label_contact: str = f'{cdir}/../consts/vertex_label_contact.npy'

############# LOSS CONFIG ##################


# TODO: Break down into parts
@dataclass
class Loss:
    type: str = 'l2'
    weight: float = 1.0

@dataclass
class Annealing:
    annealing: bool = False
    ann_end_v: float = 1.0
    e_start: int = 1
    e_end: int = 1

@dataclass
class EdgeLoss(Loss):
    type: str = 'vertex-edge'
    norm_type: str = 'l2'
    gt_edge_path: str = ''
    est_edge_path: str = ''

@dataclass
class VerticesHD(Loss):
    hd_fname: str = ''

@dataclass
class LossConfig:
    edge: Loss = Loss(type='l1', weight=0)
    vertices: Loss = Loss(type='l1', weight= 10)
    vertices_consist: Loss = Loss(type='l1', weight=0)

    rh_vertices: Loss = Loss(type='l1', weight= 7)
    feet_vertices: Loss = Loss(type='l1', weight= 3)
    pose: Loss = Loss(type='l2', weight= 10)

    vertices_hd: VerticesHD = VerticesHD(type='masked-l2')
    velocity: Loss = Loss(type='l2')
    acceleration: Loss = Loss(type='l2')

    contact: Loss = Loss(type='l1', weight= 10)

    dist_loss_exp: bool = False
    dist_loss_exp_v: bool = 16

    kl_loss: Loss = Loss(type='l1', weight=0)

    rh_faces: str = f'{cdir}/../consts/rhand_faces.npy'
    vpe_path: str = f'{cdir}/../consts/verts_per_edge_rh.npy'
    vpe_b_path: str = f'{cdir}/../consts/verts_per_edge_body.npy'
    c_weights_path: str = f'{cdir}/../consts/rhand_weight.npy'
    rh2smplx_idx: str = f'{cdir}/../consts/rhand_smplx_ids.npy'


#################### NETWORK CONFIG ########################


@dataclass
class LeakyReLU:
    negative_slope: float = 0.01


@dataclass
class ELU:
    alpha: float = 1.0


@dataclass
class PReLU:
    num_parameters: int = 1
    init: float = 0.25


@dataclass
class Activation:
    type: str = 'relu'
    inplace: bool = True

    leaky_relu: LeakyReLU = LeakyReLU()
    prelu: PReLU = PReLU()
    elu: ELU = ELU()


@dataclass
class BatchNorm:
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True


@dataclass
class GroupNorm:
    num_groups: int = 32
    eps: float = 1e-05
    affine: bool = True


@dataclass
class LayerNorm:
    eps: float = 1e-05
    elementwise_affine: bool = True


@dataclass
class Normalization:
    type: str = 'batch-norm'
    batch_norm: BatchNorm = BatchNorm()
    layer_norm = LayerNorm = LayerNorm()
    group_norm: GroupNorm = GroupNorm()

@dataclass
class LrScheduler:
    type: str = 'ReduceLROnPlateau'
    verbose: bool = True
    patience:int = 16

@dataclass
class EarlyStopping:
    monitor: str = 'val_loss'
    min_delta: float =  0.0
    patience: int =  16
    verbose: bool =  True
    mode: str = 'min'

@dataclass
class MNet:
    n_neurons: int = 2048
    dec_in: int = 7543
    out_frames: int = 10
    drop_out: float = 0.3


@dataclass
class Network:
    type: str = 'MNet'
    use_sync_bn: bool = True
    n_out_frames: int = 10
    previous_frames: int = 5
    mnet_model: MNet = MNet()
    early_stopping: EarlyStopping = EarlyStopping()
    lr_scheduler: LrScheduler = LrScheduler()



################# BODY CONFIG ######################

@dataclass
class BodyModel:
    type: str = 'smplx'
    model_path: str = ''

############## OPTIM CONFIG ######################


@dataclass
class SGD:
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class ADAM:
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    amsgrad: bool = False


@dataclass
class RMSProp:
    alpha: float = 0.99


@dataclass
class Scheduler:
    type: str = 'none'
    gamma: float = 0.1
    milestones: Optional[Tuple[int]] = tuple()
    step_size: int = 1000
    warmup_factor: float = 1.0e-1 / 3
    warmup_iters: int = 500
    warmup_method: str = 'linear'


@dataclass
class OptimConfig:
    type: str = 'adam'
    lr: float = 1e-4
    gtol: float = 1e-8
    ftol: float = -1.0
    maxiters: int = 100
    num_epochs: int = 300
    step: int = 30000
    weight_decay: float = 0.0
    weight_decay_bias: float = 0.0
    bias_lr_factor: float = 1.0

    sgd: SGD = SGD()
    adam: ADAM = ADAM()
    rmsprop: RMSProp = RMSProp()

    scheduler: Scheduler = Scheduler()


@dataclass
class Config:

    description: str = ''
    num_gpus: int = 1
    local_rank: int = 0
    use_cuda: bool = True
    is_training: bool = True
    logger_level: str = 'info'
    use_half_precision: bool = False
    pretrained: str = ''

    predict_offsets: bool = True
    use_exp: float = 5

    debug: bool = False
    seed: int = 3407
    cuda_id: int = 0

    chunk_size: int = 21
    n_epochs: int = 300

    output_folder: str = f'trained_models'
    work_dir: str = f'trained_models'
    results_base_dir: str = 'results'

    expr_ID: str = 'V00_00'

    summary_folder: str = 'summaries'
    results_folder: str = 'results'
    code_folder: str = 'code'
    best_model: Optional[str] = None

    summary_steps: int = 100
    backend: str = 'nccl'

    checkpoint_folder: str = 'checkpoints'
    checkpoint_steps: int = 1000

    eval_steps: int = 500

    float_dtype: str = 'float32'
    max_duration: float = float('inf')
    max_iters: float = float('inf')

    network: Network = Network()
    optim: OptimConfig = OptimConfig()
    body_model: BodyModel = BodyModel()
    datasets: DatasetConfig = DatasetConfig()
    losses: LossConfig = LossConfig()
    evaluation: Evaluation = Evaluation()

conf = OmegaConf.structured(Config)