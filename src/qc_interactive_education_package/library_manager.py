import json
import os
import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2
from qiskit.quantum_info import Statevector
from qiskit import transpile


class LibraryManager:
    """
    A utility class to dynamically inject quantum algorithms and mathematical
    challenges into the JSON curriculum database.
    """

    def __init__(self, filepath="library.json"):
        self.filepath = filepath
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {self.filepath} is corrupted. Initializing new database.")

        # Default empty architecture
        return {"algorithms": {}, "challenges": {}}

    def _save_data(self):
        with open(self.filepath, 'w') as f:
            # indent=4 ensures the JSON is easily readable by humans
            json.dump(self.data, f, indent=4)
        print(f"✅ Successfully updated '{self.filepath}'.")

    def _serialize_complex_array(self, state_array):
        """
        Transforms a 1D array of complex numbers into a 2D array of [real, imaginary] floats
        to strictly comply with JSON serialization standards.
        """
        serialized = []
        for amp in state_array:
            c = complex(amp)
            # Rounding to 10 decimal places prevents massive JSON file bloat from floating-point noise
            serialized.append([round(c.real, 10), round(c.imag, 10)])
        return serialized

    def add_algorithm(self, name: str, qc: QuantumCircuit):
        """
        Transpiles a Qiskit QuantumCircuit into a universal basis set, compiles it
        to OpenQASM 2.0, and saves it to the algorithm library.
        """
        try:
            # 1. Define the universal basis set that perfectly matches your UI
            # and the standard OpenQASM 2.0 specification.
            safe_basis = ['h', 'x', 'y', 'z', 'p', 'rx', 'ry', 'rz', 'cx', 'cz', 'cp', 'ccx', 'swap', 'id']

            # 2. Unroll advanced composite gates into the safe basis.
            # optimization_level=0 ensures we do not mathematically compress or alter
            # the visual step-by-step structure of the educational circuit.
            transpiled_qc = transpile(qc, basis_gates=safe_basis, optimization_level=0)

            # 3. Serialize the mathematically flattened circuit
            qasm_str = qiskit.qasm2.dumps(transpiled_qc)

            # 4. Inject and save
            self.data["algorithms"][name] = qasm_str
            self._save_data()
            print(f"✅ Algorithm '{name}' mathematically unrolled and added successfully.")

        except Exception as e:
            print(f"❌ Failed to transpile and serialize circuit to QASM: {e}")

    def add_challenge(self, name: str, initial_state: list, target_state: list):
        """
        Normalizes and injects a custom challenge directly into the database.
        """
        try:
            # Enforce mathematical normalization via Qiskit's Statevector
            sv_init = Statevector(initial_state)
            sv_targ = Statevector(target_state)

            if sv_init.num_qubits != sv_targ.num_qubits:
                raise ValueError("Dimension mismatch between initial and target states.")

            self.data["challenges"][name] = {
                "num_qubits": sv_init.num_qubits,
                "initial_state": self._serialize_complex_array(sv_init.data),
                "target_state": self._serialize_complex_array(sv_targ.data)
            }
            self._save_data()
            print(f"Challenge '{name}' added successfully.")
        except Exception as e:
            print(f"❌ Failed to add challenge: {e}")

    def add_challenge_from_circuit(self, name: str, qc: QuantumCircuit, initial_state=None):
        """
        A highly efficient method for educators: Provide a circuit, and the engine will
        automatically simulate the output tensor to define the target statevector.
        """
        try:
            # Default to standard |0...0> ground state if no initial state is provided
            if initial_state is None:
                dim = 2 ** qc.num_qubits
                initial_state = [1.0] + [0.0] * (dim - 1)

            # Initialize a fresh circuit to simulate the transformation
            sim_qc = QuantumCircuit(qc.num_qubits)
            sim_qc.initialize(initial_state, sim_qc.qubits)
            sim_qc = sim_qc.compose(qc)

            # Extract the rigorous mathematical result
            target_state = Statevector.from_instruction(sim_qc).data.tolist()

            self.add_challenge(name, initial_state, target_state)
        except Exception as e:
            print(f"❌ Failed to generate challenge from circuit: {e}")