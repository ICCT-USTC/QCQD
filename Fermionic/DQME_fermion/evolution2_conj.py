from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import RZGate, PhaseGate
from qiskit.quantum_info import Statevector
from qiskit import Aer, transpile

from aux import sigma_id, sigma_x, sigma_y, sigma_z
from aux import hs_product, kron_list, ind2state, state2paulistr, paulistr2state
from encode import JWT, doub_ani_op

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product

class Evolution():
    """
    the algorithm is inspired by the article "Quantum simulation of the Lindblad equation using a unitary decomposition of operators" 
    digital simulation of DQME equation
    """
    def __init__(self,
            sys_qubits:int,
            diss_qubits:int,
            modes:int,
            H:dict,
            gamma:np.ndarray,
            eta:np.ndarray,
            zeta:np.ndarray,
            epsilon:float,
            tau:float
        ):
        # params for model
        if not(sys_qubits == diss_qubits): 
            raise Exception('system qubits do not match diss_qubits') 
        self.sys_qubits = sys_qubits
        self.diss_qubits = diss_qubits
        self.modes = modes
        self.H = H
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta

        # params for evolution
        self.epsilon = epsilon
        self.tau = tau
        self.step = self.epsilon*self.tau

        # quantumcircuits
        self.total_qubits = sys_qubits*2+diss_qubits*modes*2+1
        self.US_circuit = None
        self.USdagger_circuit = None
        self.UA_circuit = None

    def paulistring_rotation(self,
            paulistring:str,
            paulistring_qubits:int,
            theta:float,
            ancilla_qubits:int=0
    ):
        """
        The circuit for the exp(-i P θ/2), where P is the Pauli term, 
        θ is the parameter.
        :param paulistring: the string for the Pauli term: e.g. "XIXY".
        :param paulistring_qubits: in fact the length of paulistring.
        :param ancilla_qubits: the control qubits needed.
        :param theta: the parameter θ in exp(-i P θ/2).
        :return: QuantumCircuit that implements exp(-i P θ/2) or control version of it.
        """
        if len(paulistring) != paulistring_qubits:
            raise Exception("Pauli string doesn't match to the quantum register")
    
        quantum_register = QuantumRegister(paulistring_qubits+ancilla_qubits)    
        rotation_circuit = QuantumCircuit(quantum_register)
        if np.abs(theta) <= 1E-10: 
            return rotation_circuit
        circuit_bracket = QuantumCircuit(quantum_register)
        pauli_idx = []

        for i in range(len(paulistring)):
            if paulistring[i] == 'I':
                continue
            elif paulistring[i] == 'Z':
                pauli_idx.append(i)
            elif paulistring[i] == 'X':
                circuit_bracket.h(quantum_register[i+ancilla_qubits])
                pauli_idx.append(i)
            elif paulistring[i] == 'Y':
                circuit_bracket.u(np.pi/2, np.pi/2, np.pi/2, quantum_register[i+ancilla_qubits])
                pauli_idx.append(i)

        # 'III' situation
        if (not(pauli_idx)) and (ancilla_qubits==0):
            return rotation_circuit
        if (not(pauli_idx)) and (ancilla_qubits==1):
            rotation_circuit.p(-theta/2, quantum_register[0])
            return rotation_circuit
        if (not(pauli_idx)) and (ancilla_qubits>1):
            mcp = PhaseGate(-theta/2).control(ancilla_qubits-1)
            rotation_circuit.append(mcp, quantum_register[0:ancilla_qubits])
            return rotation_circuit

        rotation_circuit.compose(circuit_bracket, quantum_register, inplace=True)

        # the first CNOTs
        for i in range(len(pauli_idx)-1):
            rotation_circuit.cx(quantum_register[pauli_idx[i]+ancilla_qubits], quantum_register[pauli_idx[i+1]+ancilla_qubits])
    
        # controlled or not controlled Rz gate
        if ancilla_qubits == 0:
            rotation_circuit.rz(theta, quantum_register[pauli_idx[-1]])
        else:
            mcRZ = RZGate(theta).control(ancilla_qubits)
            rotation_circuit.append(mcRZ, quantum_register[0:ancilla_qubits]+[quantum_register[pauli_idx[-1]+ancilla_qubits]])
    
        # the second CNOTs
        for i in reversed(range(len(pauli_idx)-1)):
            rotation_circuit.cx(quantum_register[pauli_idx[i]+ancilla_qubits], quantum_register[pauli_idx[i+1]+ancilla_qubits])
        
        rotation_circuit.compose(circuit_bracket, quantum_register, inplace=True)

        return rotation_circuit
    
    def hermi_evol(self,
            hermitian_dict:dict, 
            hermitian_qubits:int,  
            step:float,
            ancilla_qubits:int=0
    ):
        """
        The implementation of exp(-iHt), where H is the hermitian operator, t is the parameter.
        :param hermitian_dict: dictionary of Pauli terms with their weights: e.g. {"XZX": 2, "ZYI": 5, "IYZ": 7}.
        :param quantum_register: QuantumRegister.
        :param ancilla_qubits: control qubits needed.
        :param hermitian_qubits: qubits for H.
        :param step: the parameter t in exp(-iHt).
        :return: QuantumCircuit that implements controlled version of exp(-iHt).
        """
        
        quantum_register = QuantumRegister(hermitian_qubits+ancilla_qubits) # double control
        evolution_circuit = QuantumCircuit(quantum_register)

        for paulistring, coeff in hermitian_dict.items():
            pauli = paulistring[::]
            evolution_circuit.compose(self.paulistring_rotation(pauli, hermitian_qubits, 2*coeff*step, ancilla_qubits), 
                                      quantum_register, inplace=True)

        return evolution_circuit.to_instruction()
    
    def obtain_sigmastates(self, hermitian_qubits:int):
        """sigma states for n qubits system""" 
        sigmastates = []
        for ind in range(4**hermitian_qubits):
            sigmastate = ind2state(ind, hermitian_qubits)
            sigmastates.append(sigmastate)
        # pauli strings in Four-digit number form for hermitian matrices of size 2^num_qubits x 2^num_qubits
        return sigmastates
    
    def hermi_to_pauli(self, hermitian:np.ndarray, hermitian_qubits:int):
        """Decompose hermitian matrix to linear combination of pauli strings."""
        hermitian_decompsition = {} # to store results
        S = [sigma_id, sigma_x, sigma_y, sigma_z]
        norm_factor = 1/(2**hermitian_qubits)
        sigmastates = self.obtain_sigmastates(hermitian_qubits) 
        for state in sigmastates:
            label = state2paulistr(state)
            decomp = norm_factor * hs_product(kron_list([S[i] for i in state]), hermitian)
            if np.abs(decomp) >= 1E-10:
                hermitian_decompsition[label] = float(decomp)
        return hermitian_decompsition
    
    def pauli_to_hermi(self, paulistr:dict, pauli_qubits:int):
        """Convert pauli strings dict to a hermitian matrix."""
        hermitian = np.zeros((2**pauli_qubits, 2**pauli_qubits), dtype=complex)
        S = [sigma_id, sigma_x, sigma_y, sigma_z]
        for paulistring, coeff in paulistr.items():
            state = paulistr2state(paulistring)
            hermitian += coeff * kron_list([S[i] for i in state])
        return hermitian
        
    def construct_UA_circuit(self):
        """construct UA circuit unit for DQME"""
        qr = QuantumRegister(self.total_qubits)
        self.UA_circuit = QuantumCircuit(qr)

        H_qubits = self.sys_qubits
        rho_qubits = H_qubits*2
        diss_qubits = self.diss_qubits
        modes = self.modes
        diss_side = diss_qubits*modes # dissipaton qubits used on one side
        sys_diss = JWT(H_qubits+diss_side)
        diss_op = JWT(diss_side)
        # H.T left
        self.UA_circuit.append(self.hermi_evol(self.H, H_qubits, -self.tau, ancilla_qubits=0), 
                               range(1, H_qubits+1)) #H.T = H
        # H right
        self.UA_circuit.append(self.hermi_evol(self.H, H_qubits, self.tau, ancilla_qubits=0), 
                               range(H_qubits+1, rho_qubits+1))
        self.UA_circuit.barrier()
        # fk
        for idx in product(range(diss_qubits), range(modes)):
            s,k = idx
            loc = s*modes+k # loc should be will defined in future
            op_right = diss_op.num_op(loc, self.gamma[0][k].imag)
            self.UA_circuit.append(self.hermi_evol(op_right, diss_side, self.tau, ancilla_qubits=0),
                                   range(self.total_qubits-diss_side, self.total_qubits))
            op_left = diss_op.num_op(loc, self.gamma[1][k].imag) #num.T = num
            self.UA_circuit.append(self.hermi_evol(op_left, diss_side, self.tau, ancilla_qubits=0),
                                   range(rho_qubits+1, self.total_qubits-diss_side))
        self.UA_circuit.barrier()    
        # Csfk
        for idx in product(range(diss_qubits), range(modes)): # 1
            s,k = idx
            loc = s*modes+k
            coeff = self.zeta[0][k]+(self.eta[0][k]/self.zeta[0][k]).conjugate()
            op = sys_diss.sing_exc_op(qubit1=s, qubit2=H_qubits+loc, coeff=coeff/2)
            use_qubits = list(range(H_qubits+1, rho_qubits+1))+list(range(
                              self.total_qubits-diss_side, self.total_qubits))
            evolution = self.hermi_evol(op, H_qubits+diss_side, self.tau, ancilla_qubits=0)
            self.UA_circuit.append(evolution, use_qubits)
        self.UA_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 2
            s,k = idx
            loc = s*modes+k
            coeff = -self.zeta[0][k]-(self.eta[1][k]/self.zeta[0][k].conjugate())
            op = doub_ani_op(H_qubits, s, diss_side, loc, coeff/2, first_left=True)
            use_qubits = list(range(1, self.total_qubits))
            evolution = self.hermi_evol(op, self.total_qubits-1, self.tau, ancilla_qubits=0)
            self.UA_circuit.append(evolution, use_qubits)
        self.UA_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 3
            s,k = idx
            loc = s*modes+k 
            coeff = self.zeta[1][k]+(self.eta[1][k]/self.zeta[1][k]).conjugate()
            op = doub_ani_op(H_qubits, s, diss_side, loc, coeff/2, first_left=False)
            use_qubits = list(range(1, self.total_qubits))
            evolution = self.hermi_evol(op, self.total_qubits-1, self.tau, ancilla_qubits=0)
            self.UA_circuit.append(evolution, use_qubits)
        self.UA_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 4
            s,k = idx
            loc = s*modes+k
            coeff = -self.zeta[1][k]-(self.eta[0][k]/self.zeta[1][k].conjugate())
            op = sys_diss.sing_exc_op(qubit1=s, qubit2=H_qubits+loc, coeff=coeff/2)
            use_qubits = list(range(1, H_qubits+1))+list(range(
                              rho_qubits+1, self.total_qubits-diss_side))
            evolution = self.hermi_evol(op, H_qubits+diss_side, self.tau, ancilla_qubits=0)
            self.UA_circuit.append(evolution, use_qubits)
        self.UA_circuit.barrier()
        
    def construct_US_circuit(self):
        """construct US and US dagger circuit unit for DQME"""
        qr = QuantumRegister(self.total_qubits)
        self.US_circuit = QuantumCircuit(qr)

        H_qubits = self.sys_qubits
        rho_qubits = H_qubits*2
        diss_qubits = self.diss_qubits
        modes = self.modes
        diss_side = diss_qubits*modes # dissipaton qubits used on one side
        sys_diss = JWT(H_qubits+diss_side)
        diss_op = JWT(diss_side)
 
        self.US_circuit.x(qr[0]) #unusual control
        self.US_circuit.p(np.pi/2-self.epsilon, qr[0]) #global phase pi/2-epsilon
        # fk
        for idx in product(range(diss_qubits), range(modes)):
            s,k = idx
            loc = s*modes+k # loc should be will defined in future
            op_right = diss_op.num_op(loc, -self.gamma[0][k].real)
            use_qubits = [0]+list(range(self.total_qubits-diss_side, self.total_qubits))
            self.US_circuit.append(self.hermi_evol(op_right, diss_side, self.step, ancilla_qubits=1),
                                   use_qubits)
            op_left = diss_op.num_op(loc, -self.gamma[1][k].real) #num.T = num
            use_qubits = [0]+list(range(rho_qubits+1, self.total_qubits-diss_side))
            self.US_circuit.append(self.hermi_evol(op_left, diss_side, self.step, ancilla_qubits=1),
                                   use_qubits)
        self.US_circuit.barrier()
        # Csfk
        for idx in product(range(diss_qubits), range(modes)): # 1
            s,k = idx
            loc = s*modes+k
            coeff = self.zeta[0][k]-(self.eta[0][k]/self.zeta[0][k]).conjugate()
            op = sys_diss.sing_exc_op(qubit1=s, qubit2=H_qubits+loc, coeff=-1j*coeff/2)
            use_qubits = [0]+list(range(H_qubits+1, rho_qubits+1))+list(range(
                             self.total_qubits-diss_side, self.total_qubits))
            evolution = self.hermi_evol(op, H_qubits+diss_side, self.step, ancilla_qubits=1)
            self.US_circuit.append(evolution, use_qubits)
        self.US_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 2
            s,k = idx
            loc = s*modes+k
            coeff = -self.zeta[0][k]+(self.eta[1][k]/self.zeta[0][k].conjugate())
            op = doub_ani_op(H_qubits, s, diss_side, loc, -1j*coeff/2, first_left=True)
            use_qubits = list(range(0, self.total_qubits))
            evolution = self.hermi_evol(op, self.total_qubits-1, self.step, ancilla_qubits=1)
            self.US_circuit.append(evolution, use_qubits)
        self.US_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 3
            s,k = idx
            loc = s*modes+k
            coeff = self.zeta[1][k]-(self.eta[1][k]/self.zeta[1][k]).conjugate()
            op = doub_ani_op(H_qubits, s, diss_side, loc, -1j*coeff/2, first_left=False)
            use_qubits = list(range(0, self.total_qubits))
            evolution = self.hermi_evol(op, self.total_qubits-1, self.step, ancilla_qubits=1)
            self.US_circuit.append(evolution, use_qubits)
        self.US_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 4
            s,k = idx
            loc = s*modes+k
            coeff = -self.zeta[1][k]+(self.eta[0][k]/self.zeta[1][k].conjugate())
            op = sys_diss.sing_exc_op(qubit1=s, qubit2=H_qubits+loc, coeff=-1j*coeff/2)
            use_qubits = list(range(0, H_qubits+1))+list(range(
                              rho_qubits+1, self.total_qubits-diss_side))
            evolution = self.hermi_evol(op, H_qubits+diss_side, self.step, ancilla_qubits=1)
            self.US_circuit.append(evolution, use_qubits)
        self.US_circuit.barrier()
        
        self.US_circuit.x(qr[0])

    def construct_USdagger_circuit(self):
        """construct US^{\dagger} circuit unit for DQME"""
        qr = QuantumRegister(self.total_qubits)
        self.USdagger_circuit = QuantumCircuit(qr)

        H_qubits = self.sys_qubits
        rho_qubits = H_qubits*2
        diss_qubits = self.diss_qubits
        modes = self.modes
        diss_side = diss_qubits*modes # dissipaton qubits used on one side
        sys_diss = JWT(H_qubits+diss_side)
        diss_op = JWT(diss_side)

        # dagger means an inversed evolution
        self.USdagger_circuit.p(self.epsilon-np.pi/2, qr[0]) #global phase epsilon-pi/2
        # fk
        for idx in product(range(diss_qubits), range(modes)):
            s,k = idx
            # loc should be will defined in future
            loc = s*modes+k
            op_right = diss_op.num_op(loc, -self.gamma[0][k].real)
            use_qubits = [0]+list(range(self.total_qubits-diss_side, self.total_qubits))
            self.USdagger_circuit.append(self.hermi_evol(op_right, diss_side, -self.step, ancilla_qubits=1),
                                   use_qubits)
            op_left = diss_op.num_op(loc, -self.gamma[1][k].real) #num.T = num
            use_qubits = [0]+list(range(rho_qubits+1, self.total_qubits-diss_side))
            self.USdagger_circuit.append(self.hermi_evol(op_left, diss_side, -self.step, ancilla_qubits=1),
                                   use_qubits)
        self.USdagger_circuit.barrier()
        # Csfk
        for idx in product(range(diss_qubits), range(modes)): # 1
            s,k = idx
            loc = s*modes+k
            coeff = self.zeta[0][k]-(self.eta[0][k]/self.zeta[0][k]).conjugate()
            op = sys_diss.sing_exc_op(qubit1=s, qubit2=H_qubits+loc, coeff=-1j*coeff/2)
            use_qubits = [0]+list(range(H_qubits+1, rho_qubits+1))+list(range(
                             self.total_qubits-diss_side, self.total_qubits))
            evolution = self.hermi_evol(op, H_qubits+diss_side, -self.step, ancilla_qubits=1)
            self.USdagger_circuit.append(evolution, use_qubits)
        self.USdagger_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 2
            s,k = idx
            loc = s*modes+k
            coeff = -self.zeta[0][k]+(self.eta[1][k]/self.zeta[0][k].conjugate())
            op = doub_ani_op(H_qubits, s, diss_side, loc, -1j*coeff/2, first_left=True)
            use_qubits = list(range(0, self.total_qubits))
            evolution = self.hermi_evol(op, self.total_qubits-1, -self.step, ancilla_qubits=1)
            self.USdagger_circuit.append(evolution, use_qubits)
        self.USdagger_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 3
            s,k = idx
            loc = s*modes+k
            coeff = self.zeta[1][k]-(self.eta[1][k]/self.zeta[1][k]).conjugate()
            op = doub_ani_op(H_qubits, s, diss_side, loc, -1j*coeff/2, first_left=False)
            use_qubits = list(range(0, self.total_qubits))
            evolution = self.hermi_evol(op, self.total_qubits-1, -self.step, ancilla_qubits=1)
            self.USdagger_circuit.append(evolution, use_qubits)
        self.USdagger_circuit.barrier()
        for idx in product(range(diss_qubits), range(modes)): # 4
            s,k = idx
            loc = s*modes+k
            coeff = -self.zeta[1][k]+(self.eta[0][k]/self.zeta[1][k].conjugate())
            op = sys_diss.sing_exc_op(qubit1=s, qubit2=H_qubits+loc, coeff=-1j*coeff/2)
            use_qubits = list(range(0, H_qubits+1))+list(range(
                              rho_qubits+1, self.total_qubits-diss_side))
            evolution = self.hermi_evol(op, H_qubits+diss_side, -self.step, ancilla_qubits=1)
            self.USdagger_circuit.append(evolution, use_qubits)
        self.USdagger_circuit.barrier()

    def construt_trotter_circuit(self, initialstate:Statevector=None):
        """construct trotter step unit for DQME"""
        qr = QuantumRegister(self.total_qubits, 'qr')
        cr = ClassicalRegister(1)
        trotter_circuit = QuantumCircuit(qr, cr)
        if initialstate:
            trotter_circuit.initialize(initialstate, qr)
        trotter_circuit.h(qr[0])
        trotter_circuit.append(self.UA_circuit.to_instruction(label='UA'), qr)
        trotter_circuit.append(self.US_circuit.to_instruction(label='US'), qr)
        trotter_circuit.append(self.USdagger_circuit.to_instruction(label='USdagger'), qr)
        trotter_circuit.ry(-np.pi/2, qr[0])
        trotter_circuit.measure(qr[0],cr[0])
        trotter_circuit.save_statevector()

        return trotter_circuit

    def dynamics_run(self, initial:Statevector, steps:int, period:int=100):
        """dynamics for DQME""" 
        self.construct_UA_circuit()
        self.construct_US_circuit()
        self.construct_USdagger_circuit()

        statevector_list = [] # to store results
        statevector_list.append(initial)
        
        simulator = Aer.get_backend('aer_simulator')
        
        for i in tqdm(range(steps)):
            if i == 0:
                qc = self.construt_trotter_circuit(initialstate=initial)
            else:
                qc = self.construt_trotter_circuit(initialstate=statevector)
            circ = transpile(qc, simulator)
        
            control_state = 'Not 0'
            while control_state == 'Not 0':
                result = simulator.run(circ, shots=1).result()
                counts = result.get_counts()
                control_state = counts.get('0','Not 0')
                if control_state != 'Not 0': 
                    statevector = result.get_statevector()
            statevector_list.append(statevector)

            # save data in this time period
            if (i+1)%period == 0:
                rank = (i+1)//period
                data = statevector_list[(rank-1)*period+1].data
                for j in range(1, period):
                    data = np.vstack((data, statevector_list[(rank-1)*period+j+1].data))
                filename = f'{rank}statevector.dat'
                np.savetxt(filename, data)

        return statevector_list

if __name__ == '__main__':
    # set up spin-boson model
    rho_dimension = 1 
    dissipaton_mode = 1
    dissipaton_cutoff = 4 # 4 excited states

    rho_circuits= rho_dimension * 2 # two times of density matrix in vectorized form
    dissipaton_circuits = int(np.ceil(np.log2(dissipaton_cutoff))) * dissipaton_mode # 向上取整

    H = {'X':1, 'Z':1}
    Q = {'Z':0.5}

    alpha = 1
    omega = 0 # γ=α+iΩ
    zeta = 0.2
    xi = 0 

    #evolution parameters
    epsilon = 0.05
    tau = 0.05

    DQME = Evolution(rho_qubits=rho_circuits, dissipaton_qubits=dissipaton_circuits,
                     modes=dissipaton_mode, cut_off=dissipaton_cutoff,
                     H=H, Q=Q, alpha=alpha, omega=omega, zeta=zeta, xi=xi,
                     epsilon=epsilon, tau=tau)
    
    DQME.construct_USdagger_circuit()

    DQME.USdagger_circuit.decompose().draw()
    plt.show()
    plt.close()