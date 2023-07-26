import random

class BrownianVar(float):
    def __new__(cls, v, vmin=None, vmax=None, step=None):
        return super().__new__(cls, v)
    def __init__(self, v, vmin, vmax, step) -> None:
        step = abs(vmax - vmin) / 2 if step > abs(vmax - vmin) / 2 else step
        # self.v    = v
        self.vmin = vmin
        self.vmax = vmax
        self.step = step

    @property
    def v(self):
        return float(self)
    def _derive(self, v):
        return BrownianVar(v, self.vmin, self.vmax, self.step)
    def reset(self):
        vNew = random.uniform(self.vmin, self.vmax)
        return BrownianVar(vNew, self.vmin, self.vmax, self.step)
    def move(self):
        for i in range(1000):
            vNew = random.gauss(self.v, self.step)
            if vNew < self.vmax and vNew > self.vmin:
                return BrownianVar(vNew, self.vmin, self.vmax, self.step)
        print(
            f"No valid perturb, uniform reset instead! "
            + f"{self.v} {self.vmin} {self.vmax} {self.step}"
        )
        return self.reset()

    def __repr__(self):
        return f"v={self.v} vmax={self.vmax} vmin={self.vmin} step={self.step}"
    def __str__(self):
        return str(self.v)

    # def _setValue(self, v):
    #     return BrownianVar(v, self.vmin, self.vmax, self.step)

class BrownianVarMC(BrownianVar):
    def __new__(cls, v, ref=None, width=None, type=None, step=None):
        return super().__new__(cls, v)
    def __init__(self, v, ref=None, width=None, type=None, step=None) -> None:
        self._ref = ref
        self._width = width
        self._type = type
        self._step = step

    @property
    def v(self):
        return float(self)
    @property
    def vmin(self):
        if self._type == "abs":
            return self._ref - self._width
        elif self._type == "abs_pos":
            return max(self._ref - self._width, 0)
        elif self._type == "rel":
            return self._ref * (1 - self._width / 100)
        elif self._type == "rel_pos":
            return max(self._ref * (1 - self._width / 100), 0)
    @property
    def vmax(self):
        if self._type == "abs":
            return self._ref + self._width
        elif self._type == "abs_pos":
            return max(self._ref + self._width, 0)
        elif self._type == "rel":
            return self._ref * (1 + self._width / 100)
        elif self._type == "rel_pos":
            return max(self._ref * (1 + self._width / 100), 0)
    @property
    def step(self):
        return (
            abs(self.vmax - self.vmin) / 2
            if self._step > abs(self.vmax - self.vmin) / 2
            else self._step
        )

    def reset(self):
        vNew = random.uniform(self.vmin, self.vmax)
        return self._derive(vNew)
    def move(self):
        for _ in range(1000):
            vNew = random.gauss(self.v, self.step)
            if vNew < self.vmax and vNew > self.vmin:
                return self._derive(vNew)
        print(
            f"No valid perturb, uniform reset instead! "
            + f"{self.v} {self.vmin} {self.vmax} {self.step}"
        )
        return self.reset()

    def _derive(self, v, ref=None):
        return self.__class__(v, ref or self._ref, self._width, self._type, self._step)
    def __repr__(self):
        return f"v={self.v} vmax={self.vmax} vmin={self.vmin} step={self.step}"


if __name__ == "__main__":
    # a = BrownianVar(10,0,20,1)
    # print(a)
    # b = a._setValue(11)
    # print(b)

    a = BrownianVarMC(10, 10, 30, "rel", 1)
    a._ref = 15
    b = a._setValue(15)
    c = a._derive(15)
    d = a._derive(15, 15)

    print(type(a), a.__repr__())
    print(type(b), b.__repr__())
    print(type(c), c.__repr__())
    print(type(d), d.__repr__())
