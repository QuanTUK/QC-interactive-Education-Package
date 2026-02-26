import sys
import subprocess
import os
import json


def launch_app(num_qubits=3, initial_state=None):
    """
    Launches the Voilà server for the interactive quantum sandbox.

    Args:
        num_qubits (int): Number of qubits for the circuit sandbox.
        initial_state (list or None): The starting statevector amplitudes.
                                      If None, defaults to |0...0>.
    """
    print(f"Initializing Quantum Sandbox Environment ({num_qubits} Qubits)...")

    # Safety Check for the notebook
    if not os.path.exists("app.ipynb"):
        print("\n❌ ERROR: Could not find 'app.ipynb' in the current directory.")
        print("Please ensure you created this file and saved it in the same folder as this script.")
        return

    # Prepare the custom environment variables for the subprocess
    custom_env = os.environ.copy()
    custom_env["VIEWER_QUBITS"] = str(num_qubits)

    # Serialize the list (or None) to a JSON string
    custom_env["VIEWER_INITIAL"] = json.dumps(initial_state)

    print("Starting local server... A browser window will open automatically once ready.")

    # Execute Voila targeting the app notebook
    command = [
        sys.executable, "-m", "voila",
        "app.ipynb",
        "--theme=light"
    ]

    try:
        # Pass the customized environment dictionary to the subprocess
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Sandbox...")


def launch_challenge(num_qubits=1, initial_state=[1, 0], target_state=[1, -1]):
    """
    Launches the Voilà server with dynamically injected quantum states.

    Args:
        num_qubits (int): Number of qubits for the circuit.
        initial_state (list): The starting statevector amplitudes.
        target_state (list): The target statevector amplitudes to reach.
    """
    print(f"Initializing Quantum Challenge Environment ({num_qubits} Qubits)...")

    # Safety Check for the challenge notebook
    if not os.path.exists("challenge.ipynb"):
        print("\n❌ ERROR: Could not find 'challenge.ipynb' in the current directory.")
        print("Please ensure you created this file and saved it in the same folder as this script.")
        return

    # 1. Prepare the custom environment variables for the subprocess
    custom_env = os.environ.copy()
    custom_env["CHALLENGE_QUBITS"] = str(num_qubits)
    # Serialize lists to JSON strings to safely pass them through the OS layer
    custom_env["CHALLENGE_INITIAL"] = json.dumps(initial_state)
    custom_env["CHALLENGE_TARGET"] = json.dumps(target_state)

    print("Starting local server... A browser window will open automatically once ready.")

    # 2. Execute Voila targeting the challenge notebook
    command = [
        sys.executable, "-m", "voila",
        "challenge.ipynb",
        "--theme=light"
    ]

    try:
        # Pass the customized environment dictionary to the subprocess
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Challenge...")


if __name__ == "__main__":
    # Example 1: Standard launch (3 qubits, starts in |000>)
    # launch_app(num_qubits=3, initial_state=None)

    # Example 2: Launch a 2-qubit sandbox starting in a specific superposition
    launch_app(num_qubits=2, initial_state=[0.707, 0, 0, 0.707])

    # Example 3: Standard 1-qubit challenge (|0> to |->)
    # launch_challenge(num_qubits=1, initial_state=[1, 0], target_state=[1, -1])

    # Example 4: To create a 2-qubit Bell State challenge, you would modify the execution below:
    # launch_challenge(
    #     num_qubits=2,
    #     initial_state=[1, 0, 0, 0],       # |00>
    #     target_state=[1, 0, 0, 1]         # 1/sqrt(2) (|00> + |11>)
    # )