import numpy as np
import Support

class Neutron:
    def __init__(self, pos: list[float], # [x, y, z]
                 v: list[float], # [vx, vy, vz]
                 spin: int, # 1 or -1
                     ): 
        self.pos = pos
        self.v = v
        self.mu = Support.mu
        self.spin = spin
    