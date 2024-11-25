def mc_OR(x1,x2):
    weights = [1,1]
    threshold =1
    return 1 if (x1 * weights[0] + x2 * weights[1]) >= threshold else 0

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
outputs = [mc_OR(x1,x2) for x1, x2 in inputs]

print(f"OR function outputs : {outputs}")

"""
OR function outputs : [0, 1, 1, 1]

"""