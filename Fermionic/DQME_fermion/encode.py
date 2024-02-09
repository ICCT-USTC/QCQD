import numpy as np

class BosonEncode():
    """Code for boson operators"""
    def __init__(self, cut_off:int):
        self.cut_off = cut_off
        self.annihilation = None
        self.creation = None
        self.num_operator = None

    def to_binarycode(self):
        operator = np.zeros((self.cut_off, self.cut_off), dtype=complex)
        for row in range(self.cut_off-1):
            col = row+1
            operator[row,col] = np.sqrt(col)
        self.annihilation = operator
        self.creation = self.annihilation.transpose()

        operator = np.zeros((self.cut_off, self.cut_off), dtype=complex)
        for row in range(self.cut_off):
            operator[row,row] = row
        self.num_operator = operator

class JWT_Encode():
    """Jordan-Wigner transform for fermion operators"""
    def __init__(self, total_qubits:int):
        self.total_qubits = total_qubits

    def creation(self, qubit:int, coeff) -> dict:
        op_dict = {}

        term1 = 'Z'*qubit+'X'+'I'*(self.total_qubits-qubit-1)
        op_dict[term1] = coeff/2

        term2 = 'Z'*qubit+'Y'+'I'*(self.total_qubits-qubit-1)
        op_dict[term2] = -1j*coeff/2
        return op_dict
    
    def annihilation(self, qubit:int, coeff) -> dict:
        op_dict = {}

        term1 = 'Z'*qubit+'X'+'I'*(self.total_qubits-qubit-1)
        op_dict[term1] = coeff/2

        term2 = 'Z'*qubit+'Y'+'I'*(self.total_qubits-qubit-1)
        op_dict[term2] = 1j*coeff/2
        return op_dict

    def num_op(self, qubit:int, coeff) -> dict:
        """fermion number operator"""
        op_dict = {}

        term1 = 'I'*self.total_qubits
        op_dict[term1] = coeff/2

        term2 = 'I'*qubit+'Z'+'I'*(self.total_qubits-qubit-1)
        op_dict[term2] = -coeff/2

        return op_dict
    
    def sing_exc_op(self, qubit1:int, qubit2:int, coeff:complex, sequential=True) -> dict:
        """
        complex single excitation operator
        qubit1 < qubit2 is default: (qubit1) tensor (qubit2)
        """
        op_dict = {}

        term1 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term1] = coeff.real/2
        term2 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term2] = coeff.real/2
        if sequential:
            term3 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term3] = coeff.imag/2
            term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term4] = -coeff.imag/2
        else:
            term3 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term3] = -coeff.imag/2
            term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term4] = coeff.imag/2

        return op_dict
    '''
    def doub_ani_op(self, qubit1:int, qubit2:int, coeff:complex, sequential=True) -> dict:
        """double anihilation operator on one side and double excitation operator on the other"""
        op_dict = {}
        if sequential:
            term1 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term1] = -coeff.real/2
            term2 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term2] = coeff.real/2
        else:
            term1 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term1] = coeff.real/2
            term2 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term2] = -coeff.real/2

        term3 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term3] = coeff.imag/2
        term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term4] = coeff.imag/2

        return op_dict
    '''
    def coul_op(self, qubit1:int, qubit2:int, coeff) -> dict:
        """coulomb operator"""
        op_dict = {}

        term1 = 'I'*self.total_qubits
        op_dict[term1] = coeff/4

        term2 = 'I'*qubit1+'Z'+'I'*(self.total_qubits-qubit1-1)
        op_dict[term2] = -coeff/4

        term3 = 'I'*qubit2+'Z'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term3] = -coeff/4

        term4 = 'I'*qubit1+'Z'+'I'*(qubit2-qubit1-1)+'Z'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term4] = coeff/4

        return op_dict

    def num_exc_op(self, qubit1:int, qubit2:int, qubit3:int, coeff) -> dict:
        """number and exicitation operator"""
        op_dict = {}

        term1 = 'I'*qubit1+'X'+'Z'*(qubit3-qubit1-1)+'X'+'I'*(self.total_qubits-qubit3-1)
        op_dict[term1] = coeff/4

        term2 = 'I'*qubit1+'Y'+'Z'*(qubit3-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit3-1)
        op_dict[term2] = coeff/4

        term3 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'I'+'Z'*(qubit3-qubit2-1)+'X'+'I'*(self.total_qubits-qubit3-1)
        op_dict[term3] = -coeff/4

        term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'I'+'Z'*(qubit3-qubit2-1)+'X'+'I'*(self.total_qubits-qubit3-1)
        op_dict[term4] = -coeff/4

        return op_dict
    
    def doub_exc_op(self, qubit1:int, qubit2:int, qubit3:int, qubit4:int, coeff) -> dict:
        """double excitation operator"""
        op_dict = {}

        term1 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(qubit3-qubit2-1)+'X'+'Z'*(qubit4-qubit3-1)+'X'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term1] = coeff/8

        term2 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(qubit3-qubit2-1)+'Y'+'Z'*(qubit4-qubit3-1)+'Y'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term2] = coeff/8

        term3 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(qubit3-qubit2-1)+'Y'+'Z'*(qubit4-qubit3-1)+'Y'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term3] = -coeff/8

        term4 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(qubit3-qubit2-1)+'X'+'Z'*(qubit4-qubit3-1)+'X'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term4] = -coeff/8

        term5 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(qubit3-qubit2-1)+'X'+'Z'*(qubit4-qubit3-1)+'Y'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term5] = coeff/8

        term6 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(qubit3-qubit2-1)+'Y'+'Z'*(qubit4-qubit3-1)+'X'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term6] = coeff/8

        term7 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(qubit3-qubit2-1)+'X'+'Z'*(qubit4-qubit3-1)+'Y'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term7] = coeff/8

        term8 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(qubit3-qubit2-1)+'Y'+'Y'*(qubit4-qubit3-1)+'X'+'I'*(self.total_qubits-qubit4-1)
        op_dict[term8] = coeff/8

        return op_dict
    
class JWT():
    """
    Jordan-Wigner transform for fermion operators.
    Order inversed 
    """
    def __init__(self, total_qubits:int):
        self.total_qubits = total_qubits

    def creation(self, qubit:int, coeff) -> dict:
        op_dict = {}

        term1 = 'I'*qubit+'X'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term1] = coeff/2

        term2 = 'I'*qubit+'Y'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term2] = -1j*coeff/2
        return op_dict
    
    def annihilation(self, qubit:int, coeff) -> dict:
        op_dict = {}

        term1 = 'I'*qubit+'X'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term1] = coeff/2

        term2 = 'I'*qubit+'Y'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term2] = 1j*coeff/2
        return op_dict

    def num_op(self, qubit:int, coeff) -> dict:
        """fermion number operator"""
        op_dict = {}

        term1 = 'I'*self.total_qubits
        op_dict[term1] = coeff/2

        term2 = 'I'*qubit+'Z'+'I'*(self.total_qubits-qubit-1)
        op_dict[term2] = -coeff/2

        return op_dict
    
    def sing_exc_op(self, qubit1:int, qubit2:int, coeff:complex, sequential=True) -> dict:
        """
        complex single excitation operator
        qubit1 < qubit2 is default: (qubit1) tensor (qubit2)
        """
        op_dict = {}

        term1 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term1] = coeff.real/2
        term2 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term2] = coeff.real/2
        if sequential:
            term3 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term3] = coeff.imag/2
            term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term4] = -coeff.imag/2
        else:
            term3 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term3] = -coeff.imag/2
            term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term4] = coeff.imag/2

        return op_dict
    
    def coul_op(self, qubit1:int, qubit2:int, coeff) -> dict:
        """coulomb operator"""
        op_dict = {}

        term1 = 'I'*self.total_qubits
        op_dict[term1] = coeff/4

        term2 = 'I'*qubit1+'Z'+'I'*(self.total_qubits-qubit1-1)
        op_dict[term2] = -coeff/4

        term3 = 'I'*qubit2+'Z'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term3] = -coeff/4

        term4 = 'I'*qubit1+'Z'+'I'*(qubit2-qubit1-1)+'Z'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term4] = coeff/4

        return op_dict
''' 
class JWT_test():
    """
    Jordan-Wigner transform for fermion operators.
    another defination for creation and annihilation.
    """
    def __init__(self, total_qubits:int):
        self.total_qubits = total_qubits

    def creation(self, qubit:int, coeff) -> dict:
        op_dict = {}

        term1 = 'I'*qubit+'X'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term1] = coeff/2

        term2 = 'I'*qubit+'Y'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term2] = 1j*coeff/2
        return op_dict
    
    def annihilation(self, qubit:int, coeff) -> dict:
        op_dict = {}

        term1 = 'I'*qubit+'X'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term1] = coeff/2

        term2 = 'I'*qubit+'Y'+'Z'*(self.total_qubits-qubit-1)
        op_dict[term2] = -1j*coeff/2
        return op_dict
    
    def num_op(self, qubit:int, coeff) -> dict:
        """fermion number operator"""
        op_dict = {}

        term1 = 'I'*self.total_qubits
        op_dict[term1] = coeff/2

        term2 = 'I'*qubit+'Z'+'I'*(self.total_qubits-qubit-1)
        op_dict[term2] = coeff/2

        return op_dict
    
    def sing_exc_op(self, qubit1:int, qubit2:int, coeff:complex, sequential=True) -> dict:
        """
        complex single excitation operator
        qubit1 < qubit2 is default: (qubit1) tensor (qubit2)
        """
        op_dict = {}

        term1 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term1] = -coeff.real/2
        term2 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term2] = -coeff.real/2
        if sequential:
            term3 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term3] = coeff.imag/2
            term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term4] = -coeff.imag/2
        else:
            term3 = 'I'*qubit1+'Y'+'Z'*(qubit2-qubit1-1)+'X'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term3] = -coeff.imag/2
            term4 = 'I'*qubit1+'X'+'Z'*(qubit2-qubit1-1)+'Y'+'I'*(self.total_qubits-qubit2-1)
            op_dict[term4] = coeff.imag/2
        return op_dict

    def coul_op(self, qubit1:int, qubit2:int, coeff) -> dict:
        """coulomb operator"""
        op_dict = {}

        term1 = 'I'*self.total_qubits
        op_dict[term1] = coeff/4

        term2 = 'I'*qubit1+'Z'+'I'*(self.total_qubits-qubit1-1)
        op_dict[term2] = coeff/4

        term3 = 'I'*qubit2+'Z'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term3] = coeff/4

        term4 = 'I'*qubit1+'Z'+'I'*(qubit2-qubit1-1)+'Z'+'I'*(self.total_qubits-qubit2-1)
        op_dict[term4] = coeff/4

        return op_dict
'''
def doub_ani_op(A_qubits, A_qubit, B_qubits, B_qubit, coeff:complex, first_left=True) -> dict:
    """JWT encode used in fermion DQME, double annihilation or creation operators on both sides"""
    I_A = 'I'*A_qubits
    # I_B = 'I'*B_qubits
    # Z_A = 'Z'*A_qubits
    Z_B = 'Z'*B_qubits
    A_Xop = 'I'*A_qubit+'X'+'Z'*(A_qubits-A_qubit-1)
    B_Xop = 'I'*B_qubit+'X'+'Z'*(B_qubits-B_qubit-1)
    A_Yop = 'I'*A_qubit+'Y'+'Z'*(A_qubits-A_qubit-1)
    B_Yop = 'I'*B_qubit+'Y'+'Z'*(B_qubits-B_qubit-1)
    op_dict = {}

    if first_left:
        term1 = A_Xop+I_A+Z_B+B_Xop
        op_dict[term1] = coeff.real/2
        term2 = A_Yop+I_A+Z_B+B_Yop
        op_dict[term2] = -coeff.real/2
        term3 = A_Xop+I_A+Z_B+B_Yop
        op_dict[term3] = -coeff.imag/2   
        term4 = A_Yop+I_A+Z_B+B_Xop
        op_dict[term4] = -coeff.imag/2
    else:
        term1 = I_A+A_Xop+B_Xop+Z_B
        op_dict[term1] = coeff.real/2
        term2 = I_A+A_Yop+B_Yop+Z_B
        op_dict[term2] = -coeff.real/2
        term3 = I_A+A_Xop+B_Yop+Z_B
        op_dict[term3] = -coeff.imag/2
        term4 = I_A+A_Yop+B_Xop+Z_B
        op_dict[term4] = -coeff.imag/2
    return op_dict

def sing_exc_op(A_qubits, A_qubit, B_qubits, B_qubit, coeff:complex, first_left=True) -> dict:
    """
    special complex single excitation operator for DQME
    qubit1 < qubit2 is default: (qubit1) tensor (qubit2)
    """
    I_A = 'I'*A_qubits
    I_B = 'I'*B_qubits
    A_Xop = 'I'*A_qubit+'X'+'Z'*(A_qubits-A_qubit-1)
    A_Yop = 'I'*A_qubit+'Y'+'Z'*(A_qubits-A_qubit-1)
    B_Xop = 'Z'*B_qubit+'X'+'I'*(B_qubits-B_qubit-1)
    B_Yop = 'Z'*B_qubit+'Y'+'I'*(B_qubits-B_qubit-1)
    op_dict = {}

    if first_left:
        term1 = A_Xop+I_A+B_Xop+I_B
        op_dict[term1] = coeff.real/2
        term2 = A_Yop+I_A+B_Yop+I_B
        op_dict[term2] = coeff.real/2
        term3 = A_Yop+I_A+B_Xop+I_B
        op_dict[term3] = coeff.imag/2   
        term4 = A_Xop+I_A+B_Yop+I_B
        op_dict[term4] = -coeff.imag/2
    else:
        term1 = I_A+A_Xop+I_B+B_Xop
        op_dict[term1] = coeff.real/2
        term2 = I_A+A_Yop+I_B+B_Yop
        op_dict[term2] = coeff.real/2
        term3 = I_A+A_Yop+I_B+B_Xop
        op_dict[term3] = coeff.imag/2
        term4 = I_A+A_Xop+I_B+B_Yop
        op_dict[term4] = -coeff.imag/2
    return op_dict

if __name__ == '__main__':
    Bpator = JWT_Encode(total_qubits=1)
    num = Bpator.num_op(0,1)
    print(num)