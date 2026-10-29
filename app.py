"""
Grover's Algorithm demo - 3 qubits, 2 marked items (IA III, Problem 19)

Run with:   streamlit run app.py
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

N_QUBITS = 3
N_ITEMS = 2 ** N_QUBITS  # 8 records
ALL_STATES = [format(i, "03b") for i in range(N_ITEMS)]


# ---------------------------------------------------------------- circuit parts
def superposition():
    """Step 1: Hadamard on every qubit -> all 8 records equally likely."""
    qc = QuantumCircuit(N_QUBITS, name="H")
    qc.h(range(N_QUBITS))
    return qc


def oracle(marked):
    """Step 2: flip the sign of each marked state.

    For a state like '101' we X the qubits that are 0, apply a CCZ (which
    flips the sign only of |111>), then X them back.
    Qiskit prints bitstrings with qubit 2 on the LEFT, qubit 0 on the RIGHT.
    """
    qc = QuantumCircuit(N_QUBITS, name="Oracle")
    for s in marked:
        zero_qubits = [q for q in range(N_QUBITS) if s[N_QUBITS - 1 - q] == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        qc.ccz(0, 1, 2)
        if zero_qubits:
            qc.x(zero_qubits)
        qc.barrier()
    return qc


def diffuser():
    """Step 3: inversion about the average (the 'boost' step)."""
    qc = QuantumCircuit(N_QUBITS, name="Diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.ccz(0, 1, 2)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def build_circuit(marked, k, measure=True):
    qc = QuantumCircuit(N_QUBITS)
    qc.compose(superposition(), inplace=True)
    qc.barrier()
    for _ in range(k):
        qc.compose(oracle(marked), inplace=True)
        qc.compose(diffuser(), inplace=True)
        qc.barrier()
    if measure:
        qc.measure_all()
    return qc


def amplitude_table(marked, k):
    """Amplitudes after each stage (this is the 'theory' table for the report)."""
    sv = Statevector(QuantumCircuit(N_QUBITS))  # starts as |000>
    rows = {}
    sv = sv.evolve(superposition())
    rows["After H"] = sv.data.real.copy()
    for i in range(1, k + 1):
        sv = sv.evolve(oracle(marked))
        rows[f"After oracle #{i}"] = sv.data.real.copy()
        sv = sv.evolve(diffuser())
        rows[f"After diffuser #{i}"] = sv.data.real.copy()
    df = pd.DataFrame(rows, index=[f"|{s}>" for s in ALL_STATES]).round(4)
    return df, sv


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Grover's Search - 3 qubits", layout="wide")
st.title("Grover's Algorithm - 3-qubit database, 2 matching records")
st.caption("Quantum Computing (23CS2037) - IA III - Problem 19")

with st.expander("New to this? Read this first (plain English)", expanded=True):
    st.markdown(
        """
**The situation:** a database has **8 records** (3 qubits can hold 8 combinations:
000, 001, ... 111). **2 of them** match what you are searching for.

**The classical way:** check records one by one - up to 8 checks.

**Grover's way (3 steps):**
1. **Superposition (H gates):** the computer looks at all 8 records at once, each with equal chance.
2. **Oracle:** secretly flips the *sign* of the 2 correct records (like putting a hidden mark on them).
3. **Diffuser:** uses that hidden mark to *boost* the correct records and *shrink* the others.

Do steps 2-3 the right number of times and when you measure, you almost always get a correct record.
For this problem that number is just **1**, and both correct records end up with 50% chance each (100% combined).
"""
    )

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
marked = st.sidebar.multiselect(
    "Pick exactly 2 records to be the 'correct' ones",
    ALL_STATES,
    default=["101", "110"],
    max_selections=2,
)
shots = st.sidebar.slider("Shots (how many times to run the circuit)", 100, 8192, 1024, step=100)

M = 2
k_formula = math.floor((math.pi / 4) * math.sqrt(N_ITEMS / M))
k = st.sidebar.number_input(
    f"Iterations (formula says {k_formula})", min_value=0, max_value=6, value=k_formula, step=1
)
st.sidebar.markdown(
    "Try changing iterations to 2 or 3 and see the result get *worse* - "
    "that is what happens if you go past the right number."
)

if len(marked) != 2:
    st.warning("Please pick exactly 2 records in the sidebar.")
    st.stop()

marked = sorted(marked)

# ---------------------------------------------------------------- Step 1: theory
st.header("1. Theory")
st.latex(r"k = \left\lfloor \frac{\pi}{4}\sqrt{\frac{N}{M}} \right\rfloor"
         rf" = \left\lfloor \frac{{\pi}}{{4}}\sqrt{{\frac{{{N_ITEMS}}}{{{M}}}}} \right\rfloor"
         rf" = {k_formula}")

theta = math.asin(math.sqrt(M / N_ITEMS))
p_theory_formula = math.sin((2 * k + 1) * theta) ** 2
st.write(
    f"Angle: sin(theta) = sqrt(M/N) = 0.5, so theta = {math.degrees(theta):.0f} degrees. "
    f"After k = {k} iteration(s) the total chance of a correct record is "
    f"sin^2((2k+1)theta) = **{p_theory_formula:.4f}**."
)

amp_df, final_sv = amplitude_table(marked, k)
st.write("Amplitude of each record after every step (this is the table for your report):")
st.dataframe(amp_df, use_container_width=True)
st.caption(
    "Look at the oracle step: only the 2 marked records turn negative. "
    "Then the diffuser makes the marked ones big and the rest shrink."
)

# ---------------------------------------------------------------- Step 2: circuit
st.header("2. The circuit")
qc_draw = build_circuit(marked, k, measure=True)
try:
    fig_c = qc_draw.draw("mpl", fold=-1)
    st.pyplot(fig_c)
except Exception:
    st.code(str(qc_draw.draw("text", fold=200)))

c1, c2, c3 = st.columns(3)
c1.metric("Qubits", N_QUBITS)
c2.metric("Gate depth", qc_draw.decompose().depth())
c3.metric("Marked records", " & ".join(marked))

# ---------------------------------------------------------------- Step 3: simulation
st.header("3. Simulation result vs theory")
theory_probs = final_sv.probabilities()  # index i <-> bitstring format(i,'03b')
theory = {s: float(theory_probs[int(s, 2)]) for s in ALL_STATES}

sim = AerSimulator()
qc_run = build_circuit(marked, k, measure=True)
counts = sim.run(transpile(qc_run, sim), shots=shots).result().get_counts()
sim_probs = {s: counts.get(s, 0) / shots for s in ALL_STATES}

compare = pd.DataFrame(
    {
        "Record": [f"|{s}>" for s in ALL_STATES],
        "Marked?": ["YES" if s in marked else "" for s in ALL_STATES],
        "Theory probability": [round(theory[s], 4) for s in ALL_STATES],
        "Simulated probability": [round(sim_probs[s], 4) for s in ALL_STATES],
        "Simulated count": [counts.get(s, 0) for s in ALL_STATES],
    }
)

left, right = st.columns([1, 1])
with left:
    st.dataframe(compare, use_container_width=True, hide_index=True)
with right:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(N_ITEMS)
    w = 0.38
    ax.bar(x - w / 2, [theory[s] for s in ALL_STATES], w, label="Theory")
    ax.bar(x + w / 2, [sim_probs[s] for s in ALL_STATES], w, label="Simulation")
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_STATES)
    ax.set_xlabel("Record")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

combined_theory = sum(theory[s] for s in marked)
combined_sim = sum(sim_probs[s] for s in marked)
max_diff = max(abs(theory[s] - sim_probs[s]) for s in ALL_STATES)

m1, m2, m3 = st.columns(3)
m1.metric("Combined probability (theory)", f"{combined_theory:.3f}")
m2.metric("Combined probability (simulation)", f"{combined_sim:.3f}")
m3.metric("Biggest difference", f"{max_diff:.3f}")

if max_diff < 0.05:
    st.success(
        f"Theory matches simulation. Both marked records ({marked[0]} and {marked[1]}) "
        f"appear with combined probability about {combined_sim:.1%}."
    )
else:
    st.info(
        "Theory and simulation differ by more than 5%. Try more shots, "
        "or check whether you changed the iteration count from the formula value."
    )

# ---------------------------------------------------------------- classical comparison
st.header("4. Quantum vs classical")
classical_avg = (N_ITEMS + 1) / (M + 1)
st.write(
    f"A classical search checks records one by one. With {M} correct records out of {N_ITEMS}, "
    f"it needs about **{classical_avg:.1f} checks on average** (up to {N_ITEMS - M + 1} in the worst case). "
    f"Grover uses **{k} oracle call(s)**. "
    "In general classical search is O(N) and Grover is O(sqrt(N/M)); "
    "with only 8 records the saving is small, but it grows quickly for bigger databases."
)
