import numpy as np

__all__ = ['Reference', 'ConstantReference', 'StepReference',
           'PoissonStepReference', 'FixedReference']


class Reference:
    def __init__(self, seed=None):
        self.history = {'t': [], 'r': []}
        self.rng = None
        self.seed(seed)
        self.reset()

    def __call__(self, t):
        value = self._get_value(t)
        self.history['t'].append(t)
        self.history['r'].append(value)

        return value

    def _get_value(self, t):
        raise NotImplementedError

    def reset(self):
        self.history = {'t': [], 'r': []}

    def get_reference_data(self, ts=None):
        if ts is None:
            return self.history['r'], self.history['t']
        else:
            return [self._get_value(t) for t in ts], ts

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)


class ConstantReference(Reference):
    def __init__(self, low, high, value=None, seed=None):
        self.low = low
        self.high = high
        self.value = value
        super().__init__(seed)

    def reset(self, value=None):
        super().reset()
        if value is not None:
            self.value = value
        else:
            self.value = self.rng.uniform(self.low, self.high)

    def _get_value(self, t):
        return self.value


class StepReference(Reference):
    def __init__(self, low, high, step_interval, seed=None):
        self.step_interval = step_interval
        self.low = low
        self.high = high
        self.values = None
        super().__init__(seed)

    def _get_value(self, t):
        t_step_idx = int(t // self.step_interval)
        if t_step_idx >= len(self.values):
            self.values.extend([self.rng.uniform(self.low, self.high) for _ in range(t_step_idx + 1 - len(self.values))])

        return self.values[t_step_idx]

    def reset(self):
        super().reset()
        self.values = [self.rng.uniform(self.low, self.high)]


class PoissonStepReference(Reference):
    def __init__(self, low, high, rate, seed=None):
        self.rate = rate
        self.low = low
        self.high = high
        self.values = None
        super().__init__(seed)

    def _get_value(self, t):
        pass


class FixedReference(Reference):
    def __init__(self, values, timestamps, seed=None):
        self.timestamps = np.array(timestamps)
        self.values = np.array(values)
        super().__init__(seed)

    def _get_value(self, t):
        closest_index = np.abs(self.timestamps - t).argmin()
        return self.values[closest_index]
