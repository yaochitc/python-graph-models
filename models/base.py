class Model(object):
    def encoder(self, inputs):
        raise NotImplementedError

    def decoder(self, state):
        raise NotImplementedError

    def __call__(self, inputs, *args, **kwargs):
        state = self.encoder(inputs)
        loss = self.decoder(state)
        return loss