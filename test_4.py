import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

lhmc = pickle.load(open('lhmc.pkl', 'rb'))

lhmc.plot(1001)

points = np.random.uniform(0, 1, (2, 1000))
pred = lhmc.predict(points)
