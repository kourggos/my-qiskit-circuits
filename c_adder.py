from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CDKMRippleCarryAdder

def c_adder(n_bits):
    adder = CDKMRippleCarryAdder(n_bits)

    # bit de controle
    reg_control = QuantumRegister(1, "c")

    # registrador A
    reg_a = QuantumRegister(n_bits, "a")

    # registrador dos 0s
    reg_0 = QuantumRegister(n_bits, "0")

    # registrador B
    reg_b = QuantumRegister(n_bits, "b")

    #carryin
    reg_cin = QuantumRegister(1, "cin")

    # registrador para o carryout do CDKMRippleCarryAdder
    reg_cout = QuantumRegister(1, "cout")

    # inicializacao do circuito
    circuito = QuantumCircuit(reg_control, reg_cin, reg_a, reg_0, reg_b, reg_cout, name="c_adder")

    for i in range(n_bits):
        circuito.ccx(reg_control[0], reg_a[i], reg_0[i])

    circuito.append(adder, reg_cin[:] + reg_0[:] + reg_b[:] + reg_cout[:])

    for i in range(n_bits):
        circuito.ccx(reg_control[0], reg_a[i], reg_0[i])

    return circuito
