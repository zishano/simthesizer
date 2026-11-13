"""Batch expressions bound to existing kernel/tensor attributes, without graph edits."""


class BatchExpression:
    def __init__(self, evaluate):
        self.evaluate = evaluate

    def __mul__(self, factor):
        return BatchExpression(lambda values: self.evaluate(values) * factor)

    __rmul__ = __mul__

    def __floordiv__(self, divisor):
        return BatchExpression(lambda values: self.evaluate(values) // divisor)


def identity(value):
    return value


class BatchBindings:
    def __init__(self, *, record=False):
        self.record = record
        self.values = {}
        self.updates = []
        self.tensor_sizes = {}

    def parameter(self, name, value):
        self.values[name] = value
        if self.record:
            return BatchExpression(lambda values: values[name])
        return value

    def assign(self, target, attribute, value, convert=identity):
        if isinstance(value, BatchExpression):
            self.updates.append((target, attribute, value, convert))
            value = value.evaluate(self.values)
        setattr(target, attribute, convert(value))

    def bind(self, values):
        # All schema/range checks happen before changing a graph attribute.
        for target, attribute, expression, convert in self.updates:
            setattr(target, attribute, convert(expression.evaluate(values)))
        self.values = values
