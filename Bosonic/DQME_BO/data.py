from qiskit.quantum_info import Statevector
import numpy as np

def rho_tilde(statevector:Statevector, total_qubits:int):
    rho_tilde =np.zeros(shape=(2**(total_qubits-1)), dtype=complex)
    for i in range(2**(total_qubits-1)):
        rho_tilde[i] = statevector.data[2*i]
    return rho_tilde

def rdo_normal(statevector:Statevector, total_qubits:int, rho_qubits:int):
    rho_tilde =np.zeros(shape=(2**(total_qubits-1)), dtype=complex)
    for i in range(2**(total_qubits-1)):
        rho_tilde[i] = statevector.data[2*i]
    rdo = rho_tilde[0:2**rho_qubits]
    normal = rdo[0]+rdo[3]
    rdo_normal = rdo/normal
    return rdo_normal