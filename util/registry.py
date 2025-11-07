import inspect


class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = {}

    def get(self, key):
        return self._module_dict.get(key, None)

    def register(self):
        def do_register(module_class):
            assert inspect.isclass(module_class)
            module_name = module_class.__name__
            assert module_name not in self._module_dict
            self._module_dict[module_name] = module_class
            return module_class

        return do_register

    def register_module(self, module_name, module_class):
        assert inspect.isclass(module_class)
        assert isinstance(module_name, str)
        assert module_name not in self._module_dict
        self._module_dict[module_name] = module_class
        return module_class

    def register_function(self, module_name):
        def do_register(module):
            assert inspect.isfunction(module)
            assert module_name not in self._module_dict
            self._module_dict[module_name] = module
            return module

        return do_register
