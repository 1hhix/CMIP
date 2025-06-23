import torch
from itertools import chain
from collections import defaultdict, Iterable
from copy import deepcopy


def load_state_dict(self, state_dict: dict) -> None:
    """
    Loads the optimizer state from a state_dict, casting tensors to the correct device and dtype.
    Args:
        self: Optimizer instance
        state_dict: Optimizer state dict (from state_dict())
    """
    # deepcopy, to be consistent with module API
    state_dict = deepcopy(state_dict)
    # Validate the state_dict
    groups = self.param_groups
    saved_groups = state_dict["param_groups"]

    if len(groups) != len(saved_groups):
        raise ValueError(
            "loaded state dict has a different number of " "parameter groups"
        )
    param_lens = (len(g["params"]) for g in groups)
    saved_lens = (len(g["params"]) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError(
            "loaded state dict contains a parameter group "
            "that doesn't match the size of optimizer's group"
        )

    # Update the state
    id_map = {
        old_id: p
        for old_id, p in zip(
            chain(*(g["params"] for g in saved_groups)),
            chain(*(g["params"] for g in groups)),
        )
    }

    def cast(param, value):
        """
        Make a deep copy of value, casting all tensors to device of param.
        Args:
            param: Parameter tensor
            value: Value to cast
        Returns:
            Casted value
        """
        if torch.is_tensor(value):
            # Floating-point types are a bit special here. They are the only ones
            # that are assumed to always match the type of params.
            if any(
                tp in type(param.data).__name__ for tp in {"Half", "Float", "Double"}
            ):
                value = value.type_as(param.data)
            value = value.to(param.device)
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    # Copy state assigned to params (and cast tensors to appropriate types).
    # State that is not assigned to params is copied as is (needed for
    # backward compatibility).
    state = defaultdict(dict)
    for k, v in state_dict["state"].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    # Update parameter groups, setting their 'params' value
    def update_group(group, new_group):
        """
        Update a parameter group with new group values, preserving 'params'.
        Args:
            group: Original parameter group
            new_group: New group values
        Returns:
            Updated group
        """
        new_group["params"] = group["params"]
        return new_group

    param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    self.__setstate__({"state": state, "param_groups": param_groups})


torch.optim.Optimizer.load_state_dict = load_state_dict
