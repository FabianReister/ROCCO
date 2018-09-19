from yaml import load_all, load, dump


class ConfigLoader:
    _config = {}

    def load(self, filename):
        try:
            with open(filename, 'r') as file:

                self._config = load(file)
                print(self._config)

                dump(self._config)
                return True
        except:
            return False

    def getConfig(self):
        return self._config