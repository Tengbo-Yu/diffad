from typing import Iterable
from torch.nn import ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from util.dist_util import rank_zero_info


class FreezeCallback:
    def __init__(self, train_bn=False):
        self.train_bn = train_bn

    def freeze_before_training(self, torch_module):
        for module_name, module in torch_module.model.items():
            should_freeze = torch_module.should_freeze(module_name)
            if should_freeze:
                rank_zero_info(f"[FreezeCallback] freezing module: {module_name}")
                self.freeze(module, self.train_bn)

    @staticmethod
    def freeze(modules, train_bn):
        modules = FreezeCallback.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                FreezeCallback.make_trainable(mod)
            else:
                FreezeCallback.freeze_module(mod)

    @staticmethod
    def flatten_modules(modules):
        if isinstance(modules, ModuleDict):
            modules = modules.values()

        if isinstance(modules, Iterable):
            _flatten_modules = []
            for m in modules:
                _flatten_modules.extend(FreezeCallback.flatten_modules(m))
            _modules = iter(_flatten_modules)
        else:
            _modules = modules.modules()

        # Capture all leaf modules as well as parent modules that have parameters directly themselves
        return [m for m in _modules if not list(m.children()) or m._parameters]

    @staticmethod
    def make_trainable(modules):
        modules = FreezeCallback.flatten_modules(modules)
        for module in modules:
            if isinstance(module, _BatchNorm):
                module.track_running_stats = True
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for param in module.parameters(recurse=False):
                param.requires_grad = True

    @staticmethod
    def freeze_module(module):
        if isinstance(module, _BatchNorm):
            module.track_running_stats = False
        # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
        for param in module.parameters(recurse=False):
            param.requires_grad = False
