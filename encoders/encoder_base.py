class EncoderBase(object):
    def _build(self, inputs, *args, **kwargs):
        """Encodes the inputs.
        Args:
            inputs: Inputs to the encoder.
            *args: Other arguments.
            **kwargs: Keyword arguments.
        Returns:
            Encoding results.
        """
        raise NotImplementedError
