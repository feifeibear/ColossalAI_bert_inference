
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

# Hack huggingface Bert ModelOutput
# Make it available to our ColoTensor
from transformers.file_utils import ModelOutput
from dataclasses import fields


def _post_init_colotensor(self):
    class_fields = fields(self)
    # Safety and consistency checks
    if len(class_fields) == 0:
        raise ValueError(f"{self.__class__.__name__} has no fields.")
    if not all(field.default is None for field in class_fields[1:]):
        raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

    first_field = getattr(self, class_fields[0].name)
    other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

    def is_tensor_with_colo(x):
        """
        Tests if `x` is a `ColoTensor` or `torch.Tensor`.
        """
        if isinstance(x, torch.Tensor):
            return True

        return isinstance(x, ColoTensor)

    if other_fields_are_none and not is_tensor_with_colo(first_field):
        if isinstance(first_field, dict):
            iterator = first_field.items()
            first_field_iterator = True
        else:
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

        # if we provided an iterator as first field and the iterator is a (key, value) iterator
        # set the associated fields
        if first_field_iterator:
            for element in iterator:
                if (not isinstance(element, (list, tuple)) or not len(element) == 2 or not isinstance(element[0], str)):
                    break
                setattr(self, element[0], element[1])
                if element[1] is not None:
                    self[element[0]] = element[1]
        elif first_field is not None:
            self[class_fields[0].name] = first_field
    else:
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v


ModelOutput.__post_init__ = _post_init_colotensor
# complete the hack


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
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow_Linear, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    linear_spec = TensorSpec(parallel_action_list)

    
    # A naive way to set parallel spec for all weights of Linear Op
    for name, p in colo_model.colo_named_parameters():
        if not isinstance(p, ColoTensor):
            continue
        if 'weight' in name and 'LayerNorm' not in name and 'ln' not in name and 'embed' not in name:
            p.set_spec(linear_spec)

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
