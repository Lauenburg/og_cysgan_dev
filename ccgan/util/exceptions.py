
class ConfigException(Exception):
    def __call__(self, *args):
        return self.__class__(*(self.args + args))
    def __str__(self):
        return ': '.join(self.args)