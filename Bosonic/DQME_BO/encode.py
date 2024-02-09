import numpy as np

class BosonEncode():
    '''Standard Binary Code for bosons'''
    def __init__(self,
        cut_off:int
        ):
        self.cut_off = cut_off

    def construction(self):
        operator = np.zeros((self.cut_off, self.cut_off),dtype=complex)
        for row in range(self.cut_off-1):
            col = row+1
            operator[row,col] = np.sqrt(col)
        self.annihilation = operator
        self.creation = self.annihilation.transpose()

        operator = np.zeros((self.cut_off, self.cut_off),dtype=complex)
        for row in range(self.cut_off):
            operator[row,row] = row
        self.num_operator = operator

if __name__ == '__main__':
    dissipator = BosonEncode(cut_off=4)
    dissipator.construction()
    #print(dissipator.annihilation)
    #print(dissipator.creation)
    print(dissipator.num_operator)