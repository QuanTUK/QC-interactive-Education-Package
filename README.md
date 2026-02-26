# QuanTUK Quantum Computing Education Package

This is a python package to get a first glimpse into the general principles of quantum computing.
This toolset may be  used by educators or lectureres to prepare materials for theier classes, and as well interessted students to get first hand-on expierience.

## Installation
For using this package You need Python version 3.13 or higher (for the interactive python viewer). We recommend installing the latest [python release](https://www.python.org/downloads/).
Having Python installed You can install this toolset using pip
```bash
pip3 install git+https://github.com/QuanTUK/QC-interactive-Education-Package.git
```
## Usage
The following codelines provide a almost minimal examples for using this package

```python3
from qc_interactive_education_package import Simulator, CircleNotation, DimensionalCircleNotation

n = 3  # No. of qubits
sim = Simulator(n)  # Create a quantum computer simulator object
sim.write(1)  # write integer 0 in binary -> |001>
sim.had(1)  # Apply hadamard gate to qubit 0

# for visualizing
sim = DimensionalCircleNotation(sim)
sim.show()
```

For more examples and some interesting insights into the new DCN visualzation we invite You to try the examples provided in the [DCN_Examples](https://github.com/QuanTUK/DCN_examples) Repository.

You can also check out our [visual introduction to quantum communication and computation](https://github.com/QuanTUK/visual_intro_to_QC)

We also include an interactive visualization. In jupyter notebooks, run the following for the interactive viewer to show up:

```python3
from qc_interactive_education_package import InteractiveDCNViewer

viewer = InteractiveDCNViewer(num_qubits=3)

# Render a large, highly-detailed 12x8 inch plot bounded inside a 900px Jupyter container
viewer.display(figsize=(12.0, 8.0), ui_width='900px')
```

There is also challenge mode:

```python3
from qc_interactive_education_package import ChallengeDCNViewer

# Initialize the student challenge: Go from |-> to |0>
viewer = ChallengeDCNViewer(
    num_qubits=1,
    initial_state=[1, -1],  # 1/sqrt(2) (|0> - |1>)
    target_state=[1, 0]  # |0>
)

viewer.display()
```

If you have the module "voila" installed, you can launch the app also with the "launch_viewer.py" using the function "launch_app".

