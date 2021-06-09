import configargparse
import copy


class Configuration:
    def __init__(self):
        self._parser = configargparse.ArgumentParser(conflict_handler='resolve')

        # options parsed by the default parser
        self._options = None

        # individual configurations for different runs
        self._configs = []

        # arguments with more than one value
        self._multivalue_args = []

        # data types
        self._data_types = {}

        self._parser.add_argument("-d", "--defaultConfig", is_config_file=True, help="Default Config file")
        self._parser.add_argument("-c", "--config", is_config_file=True, help="Config file")

    def add_entry(self, short: str, long: str, help: str, type=str, nargs='?', default=None):
        self._parser.add("-" + short, "--" + long, help=help, is_config_file=False, type=type, nargs=nargs,
                         default=None)
        self._data_types[long.lower()] = type

    def add_model_entry(self, long: str, help: str, type=str, nargs='?'):
        self._parser.add("--" + long, help=help, is_config_file=False, type=type, nargs=nargs)
        self._data_types[long.lower()] = type

    def parse(self):
        self._options = self._parser.parse_args()

        # find values with more than one entry
        dict_options = vars(self._options)
        for k in dict_options:
            if isinstance(dict_options[k], list):
                self._multivalue_args.append(k)

        self._configs.append(self._options)
        for ma in self._multivalue_args:
            new_configs = []

            # in each config
            for c in self._configs:
                # split each attribute with multiple values
                for v in dict_options[ma]:
                    current = copy.deepcopy(c)
                    setattr(current, ma, v)
                    new_configs.append(current)

            # store splitted values
            self._configs = new_configs

    def get_configs(self):
        return self._configs
