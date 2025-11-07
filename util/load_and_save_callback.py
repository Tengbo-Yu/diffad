import os
import torch
import logging
from collections import OrderedDict


class LoadAndSaveCallback:
    def __init__(self, load_from, ignore_keys):
        self.load_from = load_from
        self.ignore_keys = ignore_keys
        self.checkpoint = None

        if load_from:
            if load_from.startswith('http'):
                self.checkpoint = torch.hub.load_state_dict_from_url(
                    load_from, map_location="cpu", progress=False
                )
            else:
                assert os.path.isfile(self.load_from)
                self.checkpoint = torch.load(self.load_from, map_location="cpu")
            print(f"[LoadAndSaveCallback] loading checkpoint from path: {self.load_from}")

    def load_diffad(self, torch_module):
        if self.checkpoint is None:
            return

        state_dict = self.checkpoint["model"]
        for module_name, module in torch_module.model.items():
            should_load = torch_module.should_load(module_name)
            if should_load:
                logging.info(f"[LoadAndSaveCallback] loading module weight: {module_name}")
                missing_keys, unexpected_keys = module.load_state_dict(
                    self.get_module_state_dict(state_dict, module_name, self.ignore_keys), strict=False)
                if len(unexpected_keys) > 0:
                    logging.warning(f"\n{'*' * 100} \n [DiffAD] unexpected keys: {unexpected_keys} \n{'*' * 100}")
                if len(missing_keys) > 0:
                    logging.warning(f"\n{'*' * 100} \n [DiffAD] missing keys: {missing_keys} \n{'*' * 100}")

    def load_ema(self, ema_module):
        if self.checkpoint is None:
            return

        state_dict = self.checkpoint["ema"]
        for module_name, module in ema_module.model.items():
            should_load = ema_module.should_load(module_name)
            if should_load:
                logging.info(f"[LoadAndSaveCallback] loading module weight: {module_name}")
                missing_keys, unexpected_keys = module.load_state_dict(
                    self.get_module_state_dict(state_dict, module_name, self.ignore_keys), strict=False)
                if len(unexpected_keys) > 0:
                    logging.warning(f"\n{'*' * 100} \n [EMA] unexpected keys: {unexpected_keys} \n{'*' * 100}")
                if len(missing_keys) > 0:
                    logging.warning(f"\n{'*' * 100} \n [EMA] missing keys: {missing_keys} \n{'*' * 100}")

    def load_opt(self, opt):
        if self.checkpoint is None:
            return
        logging.info(f"[LoadAndSaveCallback] loading module weight: optimizer")
        opt.load_state_dict(self.checkpoint['opt'])

    @staticmethod
    def get_module_state_dict(state_dict, module_name, ignore_keys):
        result = OrderedDict()
        prefix = f"model.{module_name}."
        for key, val in state_dict.items():
            if any(drop_key in key for drop_key in ignore_keys):
                continue
            if key.startswith(prefix):
                result[key[len(prefix):]] = val
        return result
