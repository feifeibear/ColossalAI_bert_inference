
import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, ColoTensor
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

from functools import partial
import random
import os
import numpy as np

from transformers import BertForSequenceClassification

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_1d_row_tp():
    
    # define a function whose return value is a model.
    def model_builder() -> torch.nn.Module:
        return BertForSequenceClassification.from_pretrained("bert-base-uncased")
    
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    # build the model using the function
    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        colo_model = model_builder()

    # define the 1D Row-wise Tensor Parallel
    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)

    
    # A naive way to set parallel spec for all weights of Linear Op
    for name, p in colo_model.colo_named_parameters():
        if not isinstance(p, ColoTensor):
            continue
        if 'weight' in name and 'LayerNorm' not in name and 'ln' not in name and 'embed' not in name:
            p.set_spec(spec)

    colo_model = colo_model.cuda()
    colo_model.eval()

    input_ids = torch.tensor(
        ([12166, 10699, 16752, 4454], [5342, 16471, 817, 16022]),
        dtype=torch.long,
        device=get_current_device())


    # make each process has the same input_ids
    torch.distributed.broadcast(input_ids, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))

    res = colo_model(input_ids).logits
    print(res.torch_tensor())


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_1d_row_tp()

def test_bert_inference(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_bert_inference(4)
