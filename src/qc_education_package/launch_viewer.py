import sys
import subprocess
import os
import json


def launch_app(num_qubits=3, initial_state=None):
    """
    Launches the Voilà server for the interactive quantum sandbox.
    Can be called from anywhere once the package is installed.
    """
    print(f"Initializing Quantum Sandbox Environment ({num_qubits} Qubits)...")

    # 1. Dynamically resolve the absolute path to the package directory
    # __file__ points to this python script. dirname gets the folder it is inside.
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the notebook
    notebook_path = os.path.join(package_dir, "app.ipynb")

    # Safety Check using the absolute path
    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'app.ipynb' at {notebook_path}")
        print("Ensure the notebook is bundled correctly in your package directory.")
        return

    # Prepare the custom environment variables for the subprocess
    custom_env = os.environ.copy()
    custom_env["VIEWER_QUBITS"] = str(num_qubits)
    custom_env["VIEWER_INITIAL"] = json.dumps(initial_state)

    print("Starting local server... A browser window will open automatically once ready.")

    # 2. Execute Voila targeting the ABSOLUTE path to the notebook
    command = [
        sys.executable, "-m", "voila",
        notebook_path,  # <--- Changed from "app.ipynb"
        "--theme=light"
    ]

    try:
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Sandbox...")


def launch_challenge(num_qubits=1, initial_state=[1, 0], target_state=[1, -1]):
    """
    Launches the Voilà server with dynamically injected quantum states.
    Can be called from anywhere once the package is installed.

    Args:
        num_qubits (int): Number of qubits for the circuit.
        initial_state (list): The starting statevector amplitudes.
        target_state (list): The target statevector amplitudes to reach.
    """
    print(f"Initializing Quantum Challenge Environment ({num_qubits} Qubits)...")

    # 1. Dynamically resolve the absolute path to the package directory
    # __file__ points to this python script. dirname gets the folder it is inside.
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the notebook
    notebook_path = os.path.join(package_dir, "challenge.ipynb")

    # Safety Check for the challenge notebook using the absolute path
    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'challenge.ipynb' at {notebook_path}")
        print("Ensure the notebook is bundled correctly in your package directory.")
        return

    # 2. Prepare the custom environment variables for the subprocess
    custom_env = os.environ.copy()
    custom_env["CHALLENGE_QUBITS"] = str(num_qubits)

    # Serialize lists to JSON strings to safely pass them through the OS layer
    custom_env["CHALLENGE_INITIAL"] = json.dumps(initial_state)
    custom_env["CHALLENGE_TARGET"] = json.dumps(target_state)

    print("Starting local server... A browser window will open automatically once ready.")

    # 3. Execute Voila targeting the ABSOLUTE path to the challenge notebook
    command = [
        sys.executable, "-m", "voila",
        notebook_path,  # <--- Changed from "challenge.ipynb"
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