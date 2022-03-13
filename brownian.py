import random

class BrownianVar(float):
    def __new__(cls,v,vmin=None,vmax=None,step=None):
        return super().__new__(cls,v)
    def __init__(self,v,vmin,vmax,step) -> None:
        step = abs(vmax-vmin)/2 if step > abs(vmax-vmin)/2 else step
        # self.v    = v
        self.vmin = vmin
        self.vmax = vmax
        self.step = step
    @property
    def v(self):
        return float(self)
    def _setValue(self,v):
        return BrownianVar(v,self.vmin,self.vmax,self.step)
    def reset(self):
        vNew = random.uniform(self.vmin,self.vmax)
        return BrownianVar(vNew,self.vmin,self.vmax,self.step)
    def move(self):
        for i in range(1000):
            vNew = random.gauss(self.v,self.step)
            if vNew < self.vmax and vNew > self.vmin:
                return BrownianVar(vNew,self.vmin,self.vmax,self.step)
        print(f'No valid perturb, uniform reset instead! '+
                f'{self.v} {self.vmin} {self.vmax} {self.step}')
        return self.reset()
    def __repr__(self):
        return f'v={self.v} vmax={self.vmax} vmin={self.vmin} step={self.step}'
    def __str__(self):
        return str(self.v)

if __name__ == '__main__':
    a = BrownianVar(10,0,20,1)
    print(a)
    b = a._setValue(11)
    print(b)