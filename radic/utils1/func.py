import torch
import torch.nn as nn
import math
import einops

def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def find_named_module(module, query):
    return next((m for (n, m) in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    return next((b for (n, b) in module.named_buffers() if n == query), None)

def _update_registered_buffer(module, buffer_name, state_dict_key, state_dict, policy='resize_if_empty', dtype=torch.int):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)
    if policy in ('resize_if_empty', 'resize'):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == 'resize' or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)
    elif policy == 'register':
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')
        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(module, module_name, buffer_names, state_dict, policy='resize_if_empty', dtype=torch.int):
    valid_buffer_names = [n for (n, _) in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
    for buffer_name in buffer_names:
        _update_registered_buffer(module, buffer_name, f'{module_name}.{buffer_name}', state_dict, policy, dtype)
