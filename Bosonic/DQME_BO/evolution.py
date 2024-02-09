from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import RZGate, PhaseGate
from qiskit.quantum_info import Statevector
from qiskit import Aer, transpile

from aux import sigma_id, sigma_x, sigma_y, sigma_z
from aux import hs_product, kron_list, ind2state, state2paulistr, paulistr2state
from encode import BosonEncode
from data import rho_tilde, rdo_normal

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Evolution():
    '''
    the algorithm is inspired by the article "Quantum simulation of the Lindblad equation using a unitary decomposition of operators" 
    digital simulation of DQME equation
    '''
    def __init__(self,
            rho_qubits:int,
            dissipaton_qubits:int,
            modes:int,
            cut_off:int,
            H:dict,
            Q:dict,
            alpha:np.ndarray,
            omega:np.ndarray,
            zeta:np.ndarray,
            xi:np.ndarray,
            epsilon:float,
            tau:float
        ):
        #params for model
        self.rho_qubits = rho_qubits
        self.dissipaton_qubits = dissipaton_qubits
        self.modes = modes
        self.cut_off = cut_off
        self.H = H
        self.Q = Q
        self.alpha = alpha
        self.omega = omega
        self.zeta = zeta
        self.xi = xi
        if (alpha.shape[0]!=modes) or (omega.shape[0]!=modes) or (zeta.shape[0]!=modes) or (xi.shape[0]!=modes):
            raise Exception("params do not match to dissipaton modes")

        #boson operator
        dissipaton = BosonEncode(cut_off=self.cut_off)
        dissipaton.construction()
        self.annihilation = dissipaton.annihilation
        self.creation = dissipaton.creation
        self.num_operator = dissipaton.num_operator
        self.boson_qubits = int(np.ceil(np.log2(self.cut_off)))

        #params for evolution
        self.epsilon = epsilon
        self.tau = tau
        self.step = self.epsilon * self.tau

        #quantumcircuits
        self.total_qubits = self.rho_qubits+self.dissipaton_qubits+1
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
        if theta == 0: 
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
    
    def hermitian_evolution(self,
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
            evolution_circuit.compose(self.paulistring_rotation(paulistring, hermitian_qubits, 2*coeff*step, ancilla_qubits), 
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
    
    def hermitian_to_paulistr(self, hermitian:np.ndarray, hermitian_qubits:int):
        """Decompose hermitian matrix to linear combination of pauli strings."""
        hermitian_decompsition = {} # to store results
        S = [sigma_id, sigma_x, sigma_y, sigma_z]
        norm_factor = 1/ (2 ** hermitian_qubits)
        sigmastates = self.obtain_sigmastates(hermitian_qubits) 
        for state in sigmastates:
            label = state2paulistr(state)
            decomp = norm_factor * hs_product(kron_list([S[i] for i in state]), hermitian)
            if np.abs(decomp) >= 1E-10:
                hermitian_decompsition[label] = float(decomp)
        return hermitian_decompsition
    
    def paulistr_to_hermitian(self, paulistr:dict, pauli_qubits:int):
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

        #hamiltonian evolution part
        H_qubits = self.rho_qubits//2
        H_matrix = self.paulistr_to_hermitian(paulistr=self.H, pauli_qubits=H_qubits)
        I = np.identity(2**H_qubits)
        H_cross = np.kron(H_matrix.transpose(), I) - np.kron(I, H_matrix)
        H_cross_dict = self.hermitian_to_paulistr(H_cross, self.rho_qubits)
        self.UA_circuit.append(self.hermitian_evolution(H_cross_dict, self.rho_qubits, -self.tau, ancilla_qubits=0),
                               qr[1:self.rho_qubits+1])
        self.UA_circuit.barrier()
        
        #dissipaton part
        #I = np.identity(2**self.boson_qubits)
        for k in range(self.modes):
            bath = self.omega[k] * (self.num_operator)
            bath_dict = self.hermitian_to_paulistr(bath, self.boson_qubits)
            self.UA_circuit.append(self.hermitian_evolution(bath_dict, self.boson_qubits, self.tau, ancilla_qubits=0),
                                   qr[self.rho_qubits+k*self.boson_qubits+1:self.rho_qubits+(k+1)*self.boson_qubits+1])
        self.UA_circuit.barrier()
            
        #interaction part
        Q_qubits = self.rho_qubits//2
        Q_matrix = self.paulistr_to_hermitian(paulistr=self.Q, pauli_qubits=Q_qubits)
        I = np.identity(2**Q_qubits)
        Q_cross = np.kron(Q_matrix.transpose(), I) - np.kron(I, Q_matrix)
        Q_cirq = np.kron(Q_matrix.transpose(), I) + np.kron(I, Q_matrix)
        for k in range(self.modes):
            first_interaction = self.zeta[k].real * np.kron(Q_cross, self.creation+self.annihilation)
            second_interaction = -0.5j * np.kron(Q_cirq, self.xi[k]*self.creation-self.xi[k].conjugate()*self.annihilation)
            interaction = first_interaction + second_interaction
            interaction_dict = self.hermitian_to_paulistr(interaction, self.rho_qubits+self.boson_qubits)
            self.UA_circuit.append(self.hermitian_evolution(interaction_dict, self.rho_qubits+self.boson_qubits, -self.tau, ancilla_qubits=0),
                                   qr[1:self.rho_qubits+1]+qr[self.rho_qubits+k*self.boson_qubits+1:self.rho_qubits+(k+1)*self.boson_qubits+1])
        self.UA_circuit.barrier()

    def construct_US_circuit(self):
        """construct US circuit unit for DQME"""
        qr = QuantumRegister(self.total_qubits)
        self.US_circuit = QuantumCircuit(qr)

        self.US_circuit.x(qr[0]) #unusual control

        self.US_circuit.p(np.pi/2-self.epsilon, qr[0]) #global phase pi/2-epsilon

        #no hamiltonian evolution part

        #dissipaton part
        #I = np.identity(2**self.boson_qubits)
        for k in range(self.modes):
            bath = self.alpha[k] * (self.num_operator)
            bath_dict = self.hermitian_to_paulistr(bath, self.boson_qubits)
            self.US_circuit.append(self.hermitian_evolution(bath_dict, self.boson_qubits, -self.step, ancilla_qubits=1),
                                   [qr[0]]+qr[self.rho_qubits+k*self.boson_qubits+1:self.rho_qubits+(k+1)*self.boson_qubits+1])
        self.US_circuit.barrier()
        
        #interaction part
        Q_qubits = self.rho_qubits//2
        Q_matrix = self.paulistr_to_hermitian(paulistr=self.Q, pauli_qubits=Q_qubits)
        I = np.identity(2**Q_qubits)
        Q_cross = np.kron(Q_matrix.transpose(), I) - np.kron(I, Q_matrix)
        Q_cirq = np.kron(Q_matrix.transpose(), I) + np.kron(I, Q_matrix)
        for k in range(self.modes):
            first_interaction = self.zeta[k].imag * np.kron(Q_cross, self.creation+self.annihilation)
            second_interaction = 0.5 * np.kron(Q_cirq, self.xi[k]*self.creation+self.xi[k].conjugate()*self.annihilation)
            interaction = first_interaction - second_interaction
            interaction_dict = self.hermitian_to_paulistr(interaction, self.rho_qubits+self.boson_qubits)
            self.US_circuit.append(self.hermitian_evolution(interaction_dict, self.rho_qubits+self.boson_qubits, -self.step, ancilla_qubits=1),
                                   qr[0:self.rho_qubits+1]+qr[self.rho_qubits+k*self.boson_qubits+1:self.rho_qubits+(k+1)*self.boson_qubits+1])
        self.US_circuit.barrier()

        self.US_circuit.x(qr[0])

    def construct_USdagger_circuit(self):
        """construct US^{\dagger} circuit unit for DQME"""
        qr = QuantumRegister(self.total_qubits)
        self.USdagger_circuit = QuantumCircuit(qr)

        #dagger means an inversed evolution
        self.USdagger_circuit.p(self.epsilon-np.pi/2, qr[0]) #global phase epsilon-pi/2

        #no hamiltonian evolution part

        #dissipaton part
        #I = np.identity(2**self.boson_qubits)
        for k in range(self.modes):
            bath = self.alpha[k] * (self.num_operator)
            bath_dict = self.hermitian_to_paulistr(bath, self.boson_qubits)
            self.USdagger_circuit.append(self.hermitian_evolution(bath_dict, self.boson_qubits, self.step, ancilla_qubits=1),
                                         [qr[0]]+qr[self.rho_qubits+k*self.boson_qubits+1:self.rho_qubits+(k+1)*self.boson_qubits+1])
        self.USdagger_circuit.barrier()

        #interaction part
        Q_qubits = self.rho_qubits//2
        Q_matrix = self.paulistr_to_hermitian(paulistr=self.Q, pauli_qubits=Q_qubits)
        I = np.identity(2**Q_qubits)
        Q_cross = np.kron(Q_matrix.transpose(), I) - np.kron(I, Q_matrix)
        Q_cirq = np.kron(Q_matrix.transpose(), I) + np.kron(I, Q_matrix)
        for k in range(self.modes): 
            first_interaction = self.zeta[k].imag * np.kron(Q_cross, self.creation+self.annihilation)
            second_interaction = 0.5 * np.kron(Q_cirq, self.xi[k]*self.creation+self.xi[k].conjugate()*self.annihilation)
            interaction = first_interaction - second_interaction
            interaction_dict = self.hermitian_to_paulistr(interaction, self.rho_qubits+self.boson_qubits)
            self.USdagger_circuit.append(self.hermitian_evolution(interaction_dict, self.rho_qubits+self.boson_qubits, self.step, ancilla_qubits=1),
                                         qr[0:self.rho_qubits+1]+qr[self.rho_qubits+k*self.boson_qubits+1:self.rho_qubits+(k+1)*self.boson_qubits+1])
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

    def dynamics_run(self, initial:Statevector, steps:int):
        """dynamics for DQME"""
        self.construct_US_circuit()
        self.construct_USdagger_circuit()
        self.construct_UA_circuit()

        statevector_list = [] # to store results
        statevector_list.append(initial)
        period = int(1/self.tau)
        
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
    #set up spin-boson model
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