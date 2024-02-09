from qiskit.quantum_info import Statevector
from typing import Union
import numpy as np

def ind2state(ind, disspaton_qubits):
    if (ind == 0): 
        return '0'*disspaton_qubits
    binary = ''
    while ind > 0:   
        binary += str(ind%2)
        ind //= 2
    if len(binary) > disspaton_qubits:
        raise Exception('illegal excited state')
    state = '0'*(disspaton_qubits-len(binary)) + binary[::-1]
    return state

def state2ind(state):
    n = len(state)
    ind = 0
    for i in range(n):
        ind += int(state[i])*(2**(n-i-1))
    return ind

def rho_tilde(statevector:Union[Statevector, np.ndarray], total_qubits:int):
    vector = statevector.data if type(statevector)==Statevector else statevector
    rho_tilde = np.zeros(shape=(2**(total_qubits-1)), dtype=complex)
    for i in range(2**(total_qubits-1)):
        rho_tilde[i] = vector[2*i]
    return rho_tilde

def rdo_anderson(rho_tilde:np.ndarray, rho_qubits:int):
    rdo_vector = rho_tilde[0:2**rho_qubits]
    rdo_T = np.reshape(rdo_vector, (2**(rho_qubits//2), 2**(rho_qubits//2)))    
    norm_factor = rdo_T.trace()
    rdo_anderson = rdo_T/norm_factor
    return rdo_anderson

def rho_tilde_normal(rho_tilde:np.ndarray, rho_qubits:int):
    rdo_vector = rho_tilde[0:2**rho_qubits]
    rdo_T = np.reshape(rdo_vector, (2**(rho_qubits//2), 2**(rho_qubits//2)))
    norm_factor = rdo_T.trace()
    rho_tilde_normal = rho_tilde/norm_factor
    return rho_tilde_normal

def rdo(rho_tilde_normal:np.ndarray, rho_qubits:int):
    rdo_vector = rho_tilde_normal[0:2**rho_qubits]
    rdo = np.reshape(rdo_vector, (2**(rho_qubits//2), 2**(rho_qubits//2))).T
    return rdo

def ddo(rho_tilde_normal:np.ndarray, rho_qubits:int, modes:int, dissipaton_qubits:int, excited_states:np.ndarray):
    if len(excited_states) != modes:
        raise Exception('illegal excited states')
    
    ID = ''
    for excited_state in reversed(excited_states):
        ID += ind2state(excited_state, dissipaton_qubits)
    ID += '0'*rho_qubits
    index = state2ind(ID)
    ddo_vector = rho_tilde_normal[index:index+2**rho_qubits]
    ddo = np.reshape(ddo_vector, (2**(rho_qubits//2), 2**(rho_qubits//2))).T
    return ddo

def observable(rho:np.ndarray, observable:np.ndarray):
    return np.dot(rho, observable).trace()

def rdo_normal(statevector:Statevector, total_qubits:int, rho_qubits:int):
    rho_tilde =np.zeros(shape=(2**(total_qubits-1)), dtype=complex)
    for i in range(2**(total_qubits-1)):
        rho_tilde[i] = statevector.data[2*i]
    rdo = rho_tilde[0:2**rho_qubits]
    normal = rdo[0]+rdo[3]
    rdo_normal = rdo/normal
    return rdo_normal

if __name__ == '__main__':
    print(ind2state(3,4))