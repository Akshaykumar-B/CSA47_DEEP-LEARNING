"""Use conditional probability to identify which machine made a defective item."""

machine_probabilities = {"Machine A": 0.60, "Machine B": 0.40}
defect_rates = {"Machine A": 0.02, "Machine B": 0.05}

defective_probability = sum(
    machine_probabilities[machine] * defect_rates[machine]
    for machine in machine_probabilities
)
posterior = {
    machine: machine_probabilities[machine] * defect_rates[machine] / defective_probability
    for machine in machine_probabilities
}

print(f"Probability an item is defective: {defective_probability:.3f}")
for machine, probability in posterior.items():
    print(f"P({machine} | defective) = {probability:.3f}")
print("A defective item is more likely to have come from Machine B because its defect rate is higher.")
