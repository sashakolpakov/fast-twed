## Time Warp Edit Distance (TWED) Library

A high-performance Python library for computing the **Time Warp Edit Distance (TWED)** 
between time series [*]. This implementation leverages **Numba** for just-in-time (JIT) 
compilation, offering significant speed-ups for dynamic programming and backtracking operations. 
The library supports both NumPy arrays and pandas DataFrames.

### Features

- **TWED Calculation:** Compute the TWED between two time series using dynamic programming.
- **Edit Path Recovery:** Retrieve the optimal sequence of operations (matches or deletions) that align the two time series.
- **Multi-format Support:** Works with both NumPy arrays and pandas DataFrames (using DataFrame indices as time stamps).
- **Performance Optimizations:**  
  - Utilizes Numba JIT for accelerated computation.
  - Minimizes per-iteration memory allocations by replacing array operations with scalar variable comparisons.
  - Optional parallelization with Numba’s `prange` for large-scale time series.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/twed-library.git
   cd twed-library
   ````
   
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
   
3. **Install the required dependencies:**
    ```bash
    pip install numpy pandas numba
    ```
   
### Example usage with Numpy arrays
```python
import numpy as np
from twed import twed 

# Create example time series data
a = np.array([[1.0], [2.0], [3.0]])
b = np.array([[1.5], [2.5], [3.5]])

# Optionally, provide custom time stamps; otherwise, default sequential indices will be used.
ts_a = np.array([0.0, 1.0, 2.0])
ts_b = np.array([0.0, 1.0, 2.0])

# Compute TWED without the edit path
distance = twed(a, b, ts_a=ts_a, ts_b=ts_b, nu=0.001, lam=1.0)
print("TWED:", distance)
```

### Example usage with Pandas DataFrames
```python
import pandas as pd
from twed import twed

# Create example DataFrames; index will be used as time stamps
df_a = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
df_b = pd.DataFrame({'value': [1.5, 2.5, 3.5]}, index=[0.0, 1.0, 2.0])

# Compute TWED and also get the optimal edit path
distance, edit_path = twed(df_a, df_b, nu=0.001, lam=1.0, path_out=True)
print("TWED:", distance)
print("Edit Path:", edit_path)
```
   
### References 

[*] Marteau, Pierre-François. "Time warp edit distance with stiffness adjustment for time series matching." IEEE transactions on pattern analysis and machine intelligence 31.2 (2008): 306-318; http://dx.doi.org/10.1109/TPAMI.2008.76.

   

