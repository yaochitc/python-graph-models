class Model(object):
    def encoder(self, inputs):
        raise NotImplementedError

    def decoder(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        pass
