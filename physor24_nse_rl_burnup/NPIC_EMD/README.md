# NPIC_EMD
Repository to house code used to generate results for the NPIC&amp;HMIT 2023 and PSA 2023 conference.

Majdi, to access surrogate models, use the `Surr` class in `surrogates.py`. Initialize the class to load the
model, then use the `predict` method.

Example:
```
import numpy as np
#depending on where you are working, may need to add repo to python path
from surrogates import Surr

s = Surr(4) #load surrogate for reactor burned for 4 years
k, Qs = s.predict(np.random.uniform(0, 360, (100, 6))) #predict k and fractional hexant powers for 100 random data points

k, Qs = s.predict(np.random.uniform(0, 360,  6)) #predict k and fractional hexant powers for 1 random data points
```
