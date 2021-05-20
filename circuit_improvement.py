import time
from copy import copy, deepcopy

import tqdm as tqdm
from pathos.multiprocessing import ProcessPool

from circuit_search_for_improvement import find_circuit
from circuit import Circuit
from itertools import combinations
import networkx as nx
import random

from circuit_search_tests import verify_sum_circuit
from functions.sum import check_sum_circuit


def correct_subcircuit_count(circuit, subcircuit_size=7, connected=True):
    circuit_graph, count = circuit.construct_graph(), 0

    for graph in (circuit_graph.subgraph(selected_nodes) for selected_nodes in
                  combinations(circuit.gates, subcircuit_size)):
        if (not connected) or (connected and nx.is_weakly_connected(graph)):
            count += 1
    return count


def make_truth_tables(circuit, subcircuit_inputs, subcircuit_outputs):
    sub_input_truth_table = {}
    sub_output_truth_table = {}
    truth_tables = circuit.get_truth_tables()

    for i in range(1 << len(circuit.input_labels)):
        str_in = [''.join(map(str, [truth_tables[g][i] for g in subcircuit_inputs]))][0]
        sub_input_truth_table[str_in] = i
        if len(sub_input_truth_table) == 1 << len(subcircuit_inputs):
            break
    sub_input_truth_table = {value: key for key, value in sub_input_truth_table.items()}

    for i in sub_input_truth_table:
        str_out = [''.join(map(str, [truth_tables[g][i] for g in subcircuit_outputs]))][0]
        sub_output_truth_table[i] = str_out
    return sub_input_truth_table, sub_output_truth_table


def make_improved_circuit_outputs(cir_out, sub_out, imp_out):
    result = list(cir_out)
    imp_out = list(imp_out)
    for index in range(0, len(result)):
        if result[index] in sub_out:
            result[index] = imp_out[sub_out.index(result[index])]
    return result


def get_inputs_and_outputs(circuit, circuit_graph, subcircuit):
    subcircuit_inputs, subcircuit_outputs = set(), set()
    for gate in subcircuit:
        for p in circuit_graph.predecessors(gate):
            if p not in subcircuit:
                subcircuit_inputs.add(p)

        if gate in circuit.outputs:
            subcircuit_outputs.add(gate)
        else:
            for s in circuit_graph.successors(gate):
                if s not in subcircuit:
                    subcircuit_outputs.add(gate)
                    break
    subcircuit_inputs = list(subcircuit_inputs)
    subcircuit_outputs = list(subcircuit_outputs)
    return subcircuit_inputs, subcircuit_outputs


# def check_subcircuit_for_improvement(curcuit, subcircuit_size, connected, subgraph):


def improve_circuit(circuit, subcircuit_size=5, connected=True):
    print('Trying to improve a circuit of size', len(circuit.gates), flush=True)
    circuit_graph = circuit.construct_graph()
    # total, current, time = correct_subcircuit_count(circuit, subcircuit_size, connected=connected), 0, 0
    # print(f'\nEnumerating subcircuits of size {subcircuit_size} (total={total})...')

    def worker(graph):
        if connected and not nx.is_weakly_connected(graph):
            return None
        subcircuit = tuple(graph.nodes)
        # start = timer()
        subcircuit_inputs, subcircuit_outputs = get_inputs_and_outputs(circuit, circuit_graph, subcircuit)
        if len(subcircuit_outputs) == subcircuit_size:
            return None
        # current += 1
        # print(f'\n{subcircuit_size}: {current}/{total} ({100 * current // total}%) ', end='', flush=True)

        random.shuffle(subcircuit_inputs)
        sub_in_tt, sub_out_tt = make_truth_tables(circuit, subcircuit_inputs, subcircuit_outputs)
        improved_circuit = find_circuit(subcircuit_inputs, subcircuit_size - 1, sub_in_tt, sub_out_tt)

        if isinstance(improved_circuit, Circuit):
            replaced_graph = circuit.replace_subgraph(improved_circuit, subcircuit, subcircuit_outputs)
            if nx.is_directed_acyclic_graph(replaced_graph):
                print('\nCircuit Improved!\n', end='', flush=True)
                improved_full_circuit = Circuit.make_circuit(replaced_graph, circuit.input_labels,
                                                             make_improved_circuit_outputs(circuit.outputs,
                                                                                           subcircuit_outputs,
                                                                                           improved_circuit.outputs))
                return fix_labels(improved_full_circuit), 1

        # stop = timer()
        # time += stop - start
        # remaining = time / current * (total - current)
        # print(f' | curr: {int(stop - start)} sec | rem: {int(remaining)} sec ({round(remaining / 60, 1)} min)', end='',
        #       flush=True)

        return None

    all_subgraphs = (circuit_graph.subgraph(selected_nodes)
                     for selected_nodes in combinations(circuit.gates, subcircuit_size))
    all_correct_subgraphs = filter(lambda gr: (not connected) or (connected and nx.is_weakly_connected(gr)),
                                   all_subgraphs)
    total = correct_subcircuit_count(circuit, subcircuit_size, connected=True)
    print("start multiprocessing")
    with ProcessPool() as pool:
        res_list = list(tqdm.tqdm(pool.imap(worker, all_correct_subgraphs), total=total))
    # res_list = [worker(gr) for gr in all_correct_subgraphs]
    res = next((item for item in res_list if item is not None), None)
    if res is not None:
        return res
    return circuit, 0



# change and improve


def random_combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def fix_labels(circuit):
    new_circuit = deepcopy(circuit)

    new_gates = {}
    fixed_gates = {}
    num = 1
    for key, _ in new_circuit.gates.items():
        new_gates[f'{key}'] = 'q' + f'{num}'
        num += 1

    num = 1
    for key in new_circuit.input_labels:
        new_gates[key] = 's' + f'{num}'
        num += 1

    for key, val in new_circuit.gates.items():
        fixed_gates[new_gates[key]] = (new_gates[val[0]], new_gates[val[1]], val[2])

    new_circuit.gates = fixed_gates

    for i in range(len(new_circuit.input_labels)):
        new_circuit.input_labels[i] = new_gates[new_circuit.input_labels[i]]

    for i in range(len(new_circuit.outputs)):
        new_circuit.outputs[i] = new_gates[new_circuit.outputs[i]]

    return new_circuit


def one_step_change_circuit(circuit, subcircuit_size=5, connected=True):
    circuit_graph = circuit.construct_graph()
    while True:
        selected_nodes = random_combination(circuit.gates, subcircuit_size)
        graph = circuit_graph.subgraph(selected_nodes)

        if connected and not nx.is_weakly_connected(graph):
            continue
        subcircuit = tuple(graph.nodes)
        subcircuit_inputs, subcircuit_outputs = get_inputs_and_outputs(circuit, circuit_graph, subcircuit)
        if len(subcircuit_outputs) == subcircuit_size:
            continue

        random.shuffle(subcircuit_inputs)
        sub_in_tt, sub_out_tt = make_truth_tables(circuit, subcircuit_inputs, subcircuit_outputs)
        improved_circuit = find_circuit(subcircuit_inputs, subcircuit_size, sub_in_tt, sub_out_tt)

        if isinstance(improved_circuit, Circuit):
            replaced_graph = circuit.replace_subgraph(improved_circuit, subcircuit, subcircuit_outputs)
            if nx.is_directed_acyclic_graph(replaced_graph):
                improved_full_circuit = Circuit.make_circuit(replaced_graph, circuit.input_labels,
                                                             make_improved_circuit_outputs(circuit.outputs,
                                                                                           subcircuit_outputs,
                                                                                           improved_circuit.outputs))
                return fix_labels(improved_full_circuit)

        continue


def change_circuit(circuit, subcircuit_size=5, connected=True, n_iter=30):
    print('Trying to change a circuit of size', len(circuit.gates),
          flush=True)
    new_circuit = circuit
    for _ in tqdm.tqdm(range(n_iter)):
        r = random.randint(0, 1)
        new_circuit = one_step_change_circuit(new_circuit, subcircuit_size - r, connected)
        assert verify_sum_circuit(new_circuit)
    return new_circuit


def improve_circuit_2(circuit, subcircuit_size=5, connected=True):
    new_circuit = change_circuit(circuit, subcircuit_size, connected, 300)
    print(new_circuit)
    check_sum_circuit(new_circuit)
    while True:
        start = time.perf_counter()
        new_circuit, flag = improve_circuit(new_circuit, subcircuit_size, connected)
        end = time.perf_counter()
        print(f"improving worked {end - start} s")
        if flag:
            print(new_circuit)
            return new_circuit
        print('Failed to improve')
        new_circuit = change_circuit(new_circuit, subcircuit_size, connected, 50)
        print(new_circuit)
