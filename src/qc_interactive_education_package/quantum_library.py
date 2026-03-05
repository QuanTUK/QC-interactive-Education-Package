import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, grover_operator
from qiskit.quantum_info import Statevector, random_statevector

class QuantumCurriculum:
    """
    A native Python constructor for the quantum education suite.
    Bypasses serialization to preserve modern Qiskit architectures,
    including advanced phase gates and custom instructions.
    """

    @staticmethod
    def get_algorithms():
        algos = {}

        # --- Algorithm 1: Bell State ---
        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        algos["Bell State Entanglement"] = qc_bell

        # --- Algorithm 2: GHZ State ---
        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)
        algos["GHZ State Entanglement"] = qc_ghz

        # --- Algorithm 3: Quantum Fourier Transform ---
        qc_qft = QuantumCircuit(3)
        qc_qft.append(QFTGate(3), [0, 1, 2])
        algos["Quantum Fourier Transform (3Q)"] = qc_qft.decompose()

        # ==========================================
        # GROVER'S SEARCH DYNAMIC BUILDER
        # ==========================================
        def build_grover(target_bitstring):
            n = len(target_bitstring)
            qc = QuantumCircuit(n)

            # 1. Initialize uniform superposition
            qc.h(range(n))

            # 2. Construct the strict single-state Oracle
            oracle = QuantumCircuit(n, name="Oracle")
            # Qiskit uses little-endian ordering (rightmost bit is q_0)
            for i, bit in enumerate(reversed(target_bitstring)):
                if bit == '0':
                    oracle.x(i)

            # Apply Multi-Controlled Phase Flip (MCZ) via Phase Kickback
            oracle.h(n - 1)
            oracle.mcx(list(range(n - 1)), n - 1)
            oracle.h(n - 1)

            # Uncompute the boundary X gates
            for i, bit in enumerate(reversed(target_bitstring)):
                if bit == '0':
                    oracle.x(i)

            # Encapsulate into a single atomic operation for the timeline
            oracle_gate = oracle.to_gate(label="Oracle")

            # 3. Construct the Diffuser (Inversion about the mean)
            diffuser = QuantumCircuit(n, name="Diffuser")
            diffuser.h(range(n))
            diffuser.x(range(n))
            diffuser.h(n - 1)
            diffuser.mcx(list(range(n - 1)), n - 1)
            diffuser.h(n - 1)
            diffuser.x(range(n))
            diffuser.h(range(n))

            # Encapsulate into a single atomic operation for the timeline
            diffuser_gate = diffuser.to_gate(label="Diffuser")

            import numpy as np
            optimal_iterations = int(np.floor((np.pi / 4.0) * np.sqrt(2 ** n)))

            for _ in range(optimal_iterations):
                qc.append(oracle_gate, range(n))
                qc.append(diffuser_gate, range(n))

            return qc

        # --- Algorithm 4 & 5: Grover's Search (Scaled) ---
        algos["Grover's Search: Target |1011⟩ (4Q)"] = build_grover("1011")
        algos["Grover's Search: Target |10101⟩ (5Q)"] = build_grover("10101")

        # --- Algorithm 6: 3-Qubit Bit-Flip Error Correction ---
        qc_3q_err = QuantumCircuit(3)
        qc_3q_err.x(0)
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)
        qc_3q_err.x(0)
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)
        qc_3q_err.ccx(1, 2, 0)
        algos["Error Correction: 3-Qubit Bit-Flip"] = qc_3q_err

        # --- Algorithm 7: 7-Qubit Steane Code (Logical |0>) ---
        qc_steane = QuantumCircuit(7)
        qc_steane.h([0, 1, 2])
        qc_steane.cx(0, 3);
        qc_steane.cx(1, 3);
        qc_steane.cx(0, 4)
        qc_steane.cx(2, 4);
        qc_steane.cx(1, 5);
        qc_steane.cx(2, 5)
        qc_steane.cx(0, 6);
        qc_steane.cx(1, 6);
        qc_steane.cx(2, 6)
        algos["Error Correction: Steane [[7,1,3]] Code"] = qc_steane

        # --- Algorithm 8: Shor's Period Finding (a=2, N=3) ---
        qc_shor = QuantumCircuit(4)
        qc_shor.h([0, 1])
        qc_shor.x(3)
        qc_shor.cswap(0, 2, 3)
        qc_shor.swap(0, 1)
        qc_shor.h(1)
        qc_shor.cp(-1*np.pi / 2, 0, 1)
        qc_shor.h(0)
        algos["Shor's Algorithm: Period Finding"] = qc_shor

        # ==========================================
        # SHOR'S PERIOD FINDING (6Q, a=5, N=6)
        # ==========================================
        qc_shor_6q = QuantumCircuit(6)
        n_count = 3

        # Step 1: Initialize Superposition & Eigenstate
        prep = QuantumCircuit(6, name="Initialize")
        for q in range(n_count):
            prep.h(q)

        # Initialize auxiliary register to |1> (Least Significant Bit of the aux register)
        prep.x(n_count)
        qc_shor_6q.append(prep.to_gate(label="Initialization"), range(6))

        # Step 2: Sequential Modular Exponentiation
        # For N=6, we need 3 work qubits.
        # We use a pure permutation matrix for x -> 5x mod 6 to guarantee mathematical perfection.
        from qiskit.circuit.library import UnitaryGate

        U_matrix = np.eye(8)
        # Map 1 to 5 and 5 to 1
        U_matrix[1, 1] = 0;
        U_matrix[1, 5] = 1
        U_matrix[5, 5] = 0;
        U_matrix[5, 1] = 1
        # Map 2 to 4 and 4 to 2
        U_matrix[2, 2] = 0;
        U_matrix[2, 4] = 1
        U_matrix[4, 4] = 0;
        U_matrix[4, 2] = 1
        # States 0, 3, 6, and 7 mathematically remain unchanged

        U_gate = UnitaryGate(U_matrix, label="5^1 mod 6").control(1)

        # The period of 5 mod 6 is 2. Therefore, U^2 and U^4 perfectly collapse to the Identity matrix.
        U_identity = UnitaryGate(np.eye(8), label="5^2 mod 6 (Identity)").control(1)
        U_identity2 = UnitaryGate(np.eye(8), label="5^4 mod 6 (Identity)").control(1)

        # Append each modular exponentiation step directly to the timeline
        qc_shor_6q.append(U_gate, [0, 3, 4, 5])
        qc_shor_6q.append(U_identity, [1, 3, 4, 5])
        qc_shor_6q.append(U_identity2, [2, 3, 4, 5])

        # Step 3: Inverse Quantum Fourier Transform
        def qft_dagger(n):
            """n-qubit QFTdagger on the first n qubits"""
            qc = QuantumCircuit(n, name="QFT†")
            for qubit in range(n // 2):
                qc.swap(qubit, n - qubit - 1)
            for j in range(n):
                for m in range(j):
                    qc.cp(-np.pi / float(2 ** (j - m)), m, j)
                qc.h(j)
            return qc.to_gate(label="Inverse QFT")

        qc_shor_6q.append(qft_dagger(n_count), range(n_count))

        algos["Shor's Algorithm: Period Finding (a=5, N=6)"] = qc_shor_6q

        # ==========================================
        # SHOR'S ALGORITHM (8Q) - SEQUENTIAL EXECUTION
        # ==========================================
        qc_shor_8q = QuantumCircuit(8)
        n_count = 4
        a = 7

        # Step 1: Initialize Superposition & Eigenstate
        prep = QuantumCircuit(8, name="Initialize")
        for q in range(n_count):
            prep.h(q)

        # Initialize auxiliary register to |1> (Least Significant Bit of the aux register)
        prep.x(n_count)
        qc_shor_8q.append(prep.to_gate(label="Initialization"), range(8))

        # Step 2: Sequential Modular Exponentiation
        def c_amod15(base, power):
            """Controlled multiplication by a mod 15"""
            if base not in [2, 4, 7, 8, 11, 13]:
                raise ValueError("'base' must be 2,4,7,8,11 or 13")
            U = QuantumCircuit(4)
            for _ in range(power):
                if base in [2, 13]:
                    U.swap(0, 1)
                    U.swap(1, 2)
                    U.swap(2, 3)
                if base in [7, 8]:
                    U.swap(2, 3)
                    U.swap(1, 2)
                    U.swap(0, 1)
                if base in [4, 11]:
                    U.swap(1, 3)
                    U.swap(0, 2)
                if base in [7, 11, 13]:
                    for q in range(4):
                        U.x(q)

            U_gate = U.to_gate()
            U_gate.name = f"{base}^{power} mod 15"
            return U_gate.control(1)

        # Append each modular exponentiation step directly to the timeline
        for q in range(n_count):
            qc_shor_8q.append(
                c_amod15(a, 2 ** q),
                [q] + [i + n_count for i in range(4)]
            )

        # Step 3: Inverse Quantum Fourier Transform
        def qft_dagger(n):
            """n-qubit QFTdagger on the first n qubits"""
            qc = QuantumCircuit(n, name="QFT†")
            for qubit in range(n // 2):
                qc.swap(qubit, n - qubit - 1)
            for j in range(n):
                for m in range(j):
                    qc.cp(-np.pi / float(2 ** (j - m)), m, j)
                qc.h(j)
            return qc.to_gate(label="Inverse QFT")

        qc_shor_8q.append(qft_dagger(n_count), range(n_count))

        algos["Shor's Algorithm: Factor 15 (8Q)"] = qc_shor_8q

        return algos

    @staticmethod
    def get_challenges():
        challenges = {}
        inv_sq2 = 1.0 / np.sqrt(2)

        challenges["Level 1: Create a Superposition (|+⟩)"] = {
            "num_qubits": 1,
            "initial_state": [1.0, 0.0],
            "target_state": [inv_sq2, inv_sq2]
        }

        challenges["Level 2: Phase Flip (|1⟩ to |-⟩)"] = {
            "num_qubits": 1,
            "initial_state": [0.0, 1.0],
            "target_state": [inv_sq2, -inv_sq2]
        }

        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        challenges["Level 3: Construct a Bell State"] = {
            "num_qubits": 2,
            "initial_state": [1.0, 0.0, 0.0, 0.0],
            "target_state": Statevector.from_instruction(qc_bell).data.tolist()
        }

        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)
        challenges["Level 4: Construct a GHZ state"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": Statevector.from_instruction(qc_ghz).data.tolist()
        }

        challenges["Level 5: Quantum Teleportation"] = {
            "num_qubits": 3,
            "initial_state": [inv_sq2, -inv_sq2, 0.0, 0.0, 0.0, 0.0, inv_sq2, -inv_sq2],
            "target_state": [inv_sq2, -inv_sq2, inv_sq2, -inv_sq2, -inv_sq2, inv_sq2, inv_sq2, -inv_sq2]
        }

        challenges["Level 6: Search Challenge: Amplify |101⟩"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": [0.0, 0.0, 0.0, 0.0, 0.0, inv_sq2, 0.0, inv_sq2]
        }

        # --- Phase-Flip Error Correction ---
        # 1. Define the pristine target: Logical |1> encoded into the phase basis (|--->)
        qc_phase_target = QuantumCircuit(3)
        qc_phase_target.x(0)
        qc_phase_target.cx(0, 1)
        qc_phase_target.cx(0, 2)
        qc_phase_target.h([0, 1, 2])

        # 2. Define the corrupted initial state: A Z-error strikes Qubit 1 (index 0)
        qc_phase_init = qc_phase_target.copy()
        qc_phase_init.z(0)

        challenges["Level 7: Correct a Phase-Flip Error"] = {
            "num_qubits": 3,
            "initial_state": Statevector.from_instruction(qc_phase_init).data.tolist(),
            "target_state": Statevector.from_instruction(qc_phase_target).data.tolist()
        }

        return challenges