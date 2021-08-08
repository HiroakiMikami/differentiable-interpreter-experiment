functions = [
    lambda xs: xs[0],  # ID
    lambda xs: xs[0] + xs[1],  # ADD
    lambda xs: xs[0] - xs[1],  # SUB
    lambda xs: xs[0] * xs[1],  # MUL
    lambda xs: xs[0] // xs[1] if xs[1] != 0 else xs[0],  # DIV
    lambda xs: xs[0] % xs[1] if xs[1] != 0 else xs[0],  # MOD
    lambda xs: int(xs[0] == xs[1]),  # EQ
    lambda xs: int(xs[0] < xs[1]),  # LT
    lambda xs: int(xs[0] > xs[1]),  # GT
    lambda xs: -xs[0],  # NEG
    lambda xs: int(not xs[0]),  # NOT
    lambda xs: int(xs[0] and xs[1]),  # AND
    lambda xs: int(xs[0] or xs[1]),  # OR
    lambda xs: xs[1] if xs[0] != 0 else xs[2],  # WHERE
]

arity = 3
