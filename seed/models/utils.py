import json
import os
import sys
from typing import OrderedDict

import torch
import torch.nn as nn

from .architectures import seed_architectures
from .simclr import SimCLR

sys.path.insert(1, os.path.dirname(__file__))
from simclr_resnet import get_resnet, name_to_params

model_file = os.path.join(os.path.dirname(__file__), "torch_hub_models.json")
torch_hub_models = json.load(open(model_file, "r"))

simclr_resnet_models = [
    f"r{d}_{w}x_sk{s}" for d in [50, 101, 152] for w in [1, 2, 3] for s in [0, 1]
]

seed_models = sorted(
    name
    for name in seed_architectures.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(seed_architectures.__dict__[name])
)

available_models = seed_models + sorted(torch_hub_models.keys()) + simclr_resnet_models

final_layer_map = {
    "resnet_moco": "fc",
    "resnet_swav": None,  # maybe "projection_head"
    "resnet_simclrv1": "fc",
    "seed_student": "fc",  # all SEED students have an FC layer?
    "resnet_simclr_clover": None,  # no FC, just load all weights that start with "encoder"
    "resnet_simclrv2": "fc",  # but it's actually under the "resnet" key
}

swav_resnet_proj_hidden_dim_map = {
    "swav_resnet50": 2048,
    "swav_resnet50w2": 8192,
    "swav_resnet50w4": 8192,
    "swav_resnet50w5": 10240,
}


def get_model_from_name(
    model_name,
    num_classes,
    pretrained=False,
    proj_out_layer=0,
    freeze_enc=True,
    init_fc=True,
):

    if init_fc:
        init_method = nn.init.kaiming_normal_
        exclude = "fc"  # TODO: implement others
    else:
        init_method = lambda x: None
        exclude = None

    # pretrained arg can be bool or string - pre-process to determine user intent:
    pretrained, ckpt_path, state_dict = get_state_dict_from_tar(
        model_name,
        pretrained,
        keep="module.encoder_q",
        exclude=exclude,
        delete_filtered_text=True,
    )

    # pretrained should be a boolean by the time it gets here
    assert isinstance(pretrained, bool), "pretrained should be a boolean"

    if model_name in seed_models:
        if "swav" in model_name:
            model = seed_architectures.__dict__[model_name](
                hidden_mlp=swav_resnet_proj_hidden_dim_map[model_name],
                output_dim=num_classes,
            )
            if state_dict is not None:
                subset_state_dict(state_dict, keep="module")
                load_state_dict_subset(model, state_dict)
            if freeze_enc:
                freeze_layers(model, ignore="", init_method=init_method)
        elif "simclr" in model_name:
            model = seed_architectures.__dict__[model_name](num_classes=num_classes)
            if state_dict is not None:
                subset_state_dict(state_dict, keep="module.module")
                load_state_dict_subset(model, state_dict)
            if freeze_enc:
                freeze_layers(model, ignore="", init_method=init_method)
            model.module.fc = nn.Linear(model.module.fc[0].in_features, num_classes)
        else:
            model = seed_architectures.__dict__[model_name](num_classes=num_classes)
            # adjust for num_classes
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            if state_dict is not None:
                load_state_dict_subset(model, state_dict)
        return model
    elif model_name in torch_hub_models:
        if state_dict or pretrained == False:
            pretrained = False  # to prevent torch.hub.load from setting pretrain=True
        model = load_torch_hub_model(
            model_name, num_classes, pretrained, freeze_enc, init_method
        )
        if state_dict is not None:
            load_state_dict_subset(model, state_dict)

        return model
    elif model_name in simclr_resnet_models:
        model, projection_head = get_resnet(*name_to_params(model_name))
        model.fc = nn.Linear(
            model.fc.in_features, num_classes
        )  # adjust for num_classes
        init_method(model.fc.weight)
        if pretrained:
            if state_dict is None:
                raise NotImplementedError(
                    f"SimCLR models don't support pretrained=True without a checkpoint."
                )
            if "resnet" in state_dict:
                model_sd = state_dict["resnet"]
                head_sd = state_dict["head"]
            elif any(k.startswith("module.encoder.net") for k in state_dict):
                model_sd = subset_state_dict(state_dict, keep="module.encoder.net")
                head_sd = subset_state_dict(state_dict, keep="module.projector")
            elif all(k.startswith("net.") or k.startswith("fc.") for k in state_dict):
                print(f"Only found resnet weights in SimCLR {model_name} checkpoint.")
                model_sd = state_dict
                head_sd = projection_head.state_dict()
            else:
                raise NotImplementedError(
                    "Please check state_dict keys to ensure they match with SimCLR resnet."
                )
            load_state_dict_subset(model, model_sd)
            load_state_dict_subset(projection_head, head_sd)

        if freeze_enc:
            freeze_layers(model, "fc", init_method)

        if proj_out_layer != 0:  # 0 means we don't want the projection head
            model = SimCLR(model, 3, 128, model.fc.in_features, proj_out_layer)
            load_state_dict_subset(model.projector, head_sd)
        return model
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")


def get_state_dict_from_tar(
    model_name,
    pretrained,
    keep="module.encoder_q",
    exclude="fc",
    delete_filtered_text=True,
):
    if isinstance(pretrained, bool):
        return pretrained, None, None
    else:
        pretrained, ckpt_path = get_pretrained_path(pretrained)
        if ckpt_path is not None:
            print(f"Loading {model_name} checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = state_dict.get("model", state_dict)
        state_dict = subset_state_dict(state_dict, keep, exclude, delete_filtered_text)
        return pretrained, ckpt_path, state_dict


def get_pretrained_path(pretrained):
    """
    The `pretrained` argument can be a bool or string.
    This function preprocesses the argument and determines whether
    the user wants to use a pretrained model or not, and
    whether the user is passing in a checkpoint/path to load from.
    """
    if isinstance(pretrained, str):
        ckpt_path = pretrained
        pretrained = pretrained.lower() not in ["false", "no", "none"]
        if not pretrained:
            ckpt_path = None
        else:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"Pretrained checkpoint file {ckpt_path} not found."
                )
    else:
        ckpt_path = pretrained
    return pretrained, ckpt_path


def subset_state_dict(
    state_dict, keep="module.encoder_q", exclude="fc", delete_filtered_text=True
):
    for k in list(state_dict.keys()):
        if keep in k:
            if delete_filtered_text:
                state_dict[k.replace(f"{keep}.", "")] = state_dict[k]
                del state_dict[k]
        elif not (exclude is None or exclude == "") and (
            f".{exclude}" in k or f"{exclude}." in k
        ):
            print(f"Deleting {k} from state_dict because {exclude=}.")
            del state_dict[k]
    return state_dict


def load_moco_weights(model, state_dict, model_name):
    if state_dict is None:
        print(f"state_dict is None... skip loading weights.")
        return
    else:
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            del state_dict[k]  # delete renamed or unused k (e.g., encoder_k)
        load_state_dict_subset(model, state_dict)


def load_torch_hub_model(model_name, num_classes, pretrained, freeze_enc, init_method):
    model = torch.hub.load(
        torch_hub_models[model_name], model_name, pretrained=pretrained
    )
    final_layer_name = None
    # adjust for num_classes
    if model_name == "dino_resnet50":
        # address issue with dino resnet where model.fc is an Identity layer
        new_resnet = torch.hub.load(
            torch_hub_models["resnet50"], "resnet50", pretrained=False
        )
        new_resnet.fc = nn.Linear(new_resnet.fc.in_features, num_classes)
        load_state_dict_subset(new_resnet, model.state_dict())
        del model  # delete old model
        model = new_resnet
        if freeze_enc:
            freeze_layers(model, "fc", init_method)
        else:
            init_method(model.fc.weight)
        final_layer_name = "fc"
    elif "resnet" in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_enc:
            freeze_layers(model, "fc", init_method)
        else:
            init_method(model.fc.weight)
        final_layer_name = "fc"
    elif "densenet" in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        if freeze_enc:
            freeze_layers(model, "classifier", init_method)
        else:
            init_method(model.classifier.weight)
        final_layer_name = "classifier"
    elif "efficientnet" in model_name:
        model.classifier[-1].out_features = num_classes
        final_layer_name = "classifier"
    elif "dino" in model_name and "vit" in model_name:
        head = nn.Linear(model.norm.normalized_shape[0], num_classes)
        model = nn.Sequential(OrderedDict({"vit": model, "head": head}))
        if freeze_enc:
            freeze_layers(model, "head", init_method)
        else:
            init_method(model.head.weight)
        final_layer_name = "head"
    elif "dino" in model_name and "xcit" in model_name:
        model.head = nn.Linear(model.num_features, num_classes)
        if freeze_enc:
            freeze_layers(model, "head", init_method)
        else:
            init_method(model.head.weight)
        final_layer_name = "head"
    else:
        raise NotImplementedError(
            f"Don't know how to adjust num_classes for {model_name}"
        )
    return model


def load_state_dict_subset(model, state: dict, verbose=True):
    # only keep weights whose shapes match the corresponding layers in the model
    if "state_dict" in state:
        state = state["state_dict"]
    elif "resnet" in state:  # simclrv2 weights split into "resnet" and "head"
        state = state["resnet"]
    elif "head" in state:
        state = state["head"]
    elif "model" in state:
        state = state["model"]
    m_s = model.state_dict()
    if not verbose:
        subset = {
            k: v for k, v in state.items() if k in m_s and m_s[k].shape == v.shape
        }
    else:
        subset = dict()
        for k, v in state.items():
            if k in m_s and m_s[k].shape == v.shape:
                subset[k] = v
            elif k not in m_s:
                print(f"{k} not in model")
            elif k in m_s and m_s[k].shape != v.shape:
                print(f"{k} in model but shape {m_s[k].shape} != {v.shape}")
            else:
                print(f"{k} in model, but couldn't load for some reason.")
    print(f"Loaded {len(subset)}/{len(state)} weights for {model.__class__.__name__}.")
    model.load_state_dict(subset, strict=False)
    return subset


def freeze_layers(model, ignore="fc", init_method=None):
    # freeze all layers except those in ignore
    if ignore != "":
        ignore_list = [f"{ignore}.weight", f"{ignore}.bias"]
    else:
        ignore_list = []
    for name, param in model.named_parameters():
        if name not in ignore_list:
            param.requires_grad = False
    # init the fc layer
    if ignore != "":
        if init_method is None:
            exec(f"model.{ignore}.weight.data.normal_(mean=0.0, std=0.01)")
            exec(f"model.{ignore}.bias.data.zero_()")
        else:
            exec(f"init_method(model.{ignore}.weight)")
            # exec(f"model.{ignore}.bias.data.zero_()")  #? Check whether we need this
