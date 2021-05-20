import argparse

from circuit_improvement import *
from functions.ex3 import *
from functions.ex2 import *
from functions.sum import *
from functions.ib import *
from functions.maj import *
from functions.mod3 import *
from functions.th import *


def run_file_improve_circuit(filename, subcircuit_size=5, connected=True):
    print(f'Run {filename}...')
    circuit = Circuit()
    circuit.load_from_file(filename)
    return improve_circuit(circuit, subcircuit_size, connected)


def run_improve_circuit(fun, input_size, subcircuit_size=5, connected=True):
    print(f'Run {fun.__name__}...')
    circuit = Circuit(input_labels=[f'x{i}' for i in range(1, input_size + 1)], gates={})
    circuit.outputs = fun(circuit, circuit.input_labels)
    if isinstance(circuit.outputs, str):
        circuit.outputs = [circuit.outputs]
    return improve_circuit(circuit, subcircuit_size, connected)


def run_change_circuit(fun, input_size, subcircuit_size=5, connected=True):
    print(f'Run {fun.__name__}...')
    circuit = Circuit(input_labels=[f'x{i}' for i in range(1, input_size + 1)], gates={})
    circuit.outputs = fun(circuit, circuit.input_labels)
    if isinstance(circuit.outputs, str):
        circuit.outputs = [circuit.outputs]
    return change_circuit(circuit, subcircuit_size, connected)


def run_imp_circuit(fun, input_size, subcircuit_size=5, connected=True) -> Circuit:
    print(f'Run {fun.__name__}...')
    circuit = Circuit(input_labels=[f'x{i}' for i in range(1, input_size + 1)], gates={})
    circuit.outputs = fun(circuit, circuit.input_labels)
    if isinstance(circuit.outputs, str):
        circuit.outputs = [circuit.outputs]
    return improve_circuit_2(circuit, subcircuit_size, connected)


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--command", type=str, default="ch")
    return parser


if __name__ == '__main__':
    args = init_parser().parse_args()
    if args.seed >= 0:
        random.seed(args.seed)
    command = args.command

    if command == 'r':
        improved_circuit = run_improve_circuit(add_th2_12_31, 12, subcircuit_size=5, connected=True)
        print(improved_circuit)
        improved_circuit.draw('sum5')
    elif command == 'rf':
        run_file_improve_circuit('sum/sum7_sub', subcircuit_size=5, connected=True)
    elif command == 'mc':
        Circuit.make_code('ex/ex2_over1_size13', 'code')
    elif command == 'd':
        c = Circuit(fn='sum/sum3_size5').draw('sum3_size5')

    elif command == 'ch':
        imp_circuit = run_imp_circuit(add_sum15_51, 15, subcircuit_size=5, connected=True)
        # imp_circuit = improve_circuit_2(imp_circuit,k
        #                               subcircuit_size=4,
        #                               connected=True)
        print(imp_circuit)
        imp_circuit.save_to_file("out_circuit")
        imp_circuit.draw('out_circuit_image')
