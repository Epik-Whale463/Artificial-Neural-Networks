def mc_AND(x1,x2):
    weights = [1,1]
    threshold = 2
    return 1 if (x1 * weights[0] + x2 * weights[1]) >= threshold else 0


# Testing AND function
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
outputs = [mc_AND(x1, x2) for x1, x2 in inputs]

print("AND Function Output:", outputs)

"""
AND Function Output: [0, 0, 0, 1]

"""