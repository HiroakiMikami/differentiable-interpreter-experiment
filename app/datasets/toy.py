from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class FunctionName(Enum):
    ID = auto()
    NEG = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    WHERE = auto()

    @staticmethod
    def arity(value) -> int:
        if value in set([FunctionName.ID, FunctionName.NEG, FunctionName.NOT]):
            return 1
        elif value == FunctionName.WHERE:
            return 3
        else:
            return 2


class Program:
    @property
    def n_input(self) -> int:
        raise NotImplementedError


@dataclass
class Input(Program):
    id: int

    def __eq__(self, other) -> bool:
        if not isinstance(other, Input):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def n_input(self) -> int:
        return self.id + 1


class Constant(Program):
    pass


@dataclass
class Number(Constant):
    value: int

    def __eq__(self, other) -> bool:
        if not isinstance(other, Number):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def n_input(self) -> int:
        return 0


@dataclass
class Boolean(Constant):
    value: bool

    def __eq__(self, other) -> bool:
        if not isinstance(other, Boolean):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def n_input(self) -> int:
        return 0


@dataclass
class Function(Program):
    name: FunctionName
    args: List[Program]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Function):
            return False
        return self.name == other.name and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.args)))

    @property
    def n_input(self) -> int:
        return max([arg.n_input for arg in self.args])


class Parser:
    def parse(self, code: str) -> Optional[Program]:
        name2enum = FunctionName.__members__
        tokens = [token.strip() for token in code.split(" ")]

        class FailToParse(Exception):
            pass

        def _parse(tokens: List[str], offset: int) -> Tuple[Program, int]:
            if tokens[offset] == "input":
                # Input
                if offset + 1 == len(tokens):
                    raise FailToParse()
                id = tokens[offset + 1]
                try:
                    return Input(int(id)), offset + 2
                except ValueError:
                    raise FailToParse()
            elif tokens[offset] in name2enum:
                # Function
                name = name2enum[tokens[offset]]
                if offset + 1 == len(tokens):
                    raise FailToParse()
                offset += 1
                if tokens[offset] != "(":
                    raise FailToParse()
                offset += 1
                args: List[Program] = []
                while offset < len(tokens):
                    token = tokens[offset]
                    if token == ")":
                        return Function(name, args), offset + 1
                    arg, offset = _parse(tokens, offset)
                    args.append(arg)
                raise FailToParse()
            elif tokens[offset] in set(["True", "False"]):
                # Boolean
                return Boolean(tokens[offset] == "True"), offset + 1
            else:
                # Number
                try:
                    return Number(int(tokens[offset])), offset + 1
                except ValueError:
                    raise FailToParse()

        try:
            p, offset = _parse(tokens, 0)
            if len(tokens) != offset:
                return None
            else:
                return p
        except FailToParse:
            return None

    def unparse(self, p: Program) -> str:
        if isinstance(p, Input):
            return f"input {p.id}"
        elif isinstance(p, Number):
            return str(p.value)
        elif isinstance(p, Boolean):
            return str(p.value)
        elif isinstance(p, Function):
            args = " ".join([self.unparse(arg) for arg in p.args])
            return f"{p.name.name} ( {args} )"
        else:
            raise AssertionError(f"Invalid program type: {p}")


class Interpreter:
    def run(self, p: Program, input: List[int]) -> Optional[Union[bool, int]]:
        class FailToRun(Exception):
            pass

        def _run(p: Program) -> Union[bool, int]:
            if isinstance(p, Input):
                if p.id < len(input):
                    return input[p.id]
                raise FailToRun()
            elif isinstance(p, Number):
                return p.value
            elif isinstance(p, Boolean):
                return p.value
            elif isinstance(p, Function):
                args = [_run(arg) for arg in p.args]
                if len(args) != FunctionName.arity(p.name):
                    raise FailToRun()

                if p.name == FunctionName.ID:
                    return args[0]
                elif p.name == FunctionName.NEG:
                    if isinstance(args[0], int):
                        return -args[0]
                    raise FailToRun()
                elif p.name == FunctionName.ADD:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        return args[0] + args[1]
                    raise FailToRun()
                elif p.name == FunctionName.SUB:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        return args[0] - args[1]
                    raise FailToRun()
                elif p.name == FunctionName.MUL:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        return args[0] * args[1]
                    raise FailToRun()
                elif p.name == FunctionName.DIV:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        if args[1] == 0:
                            raise FailToRun()
                        return args[0] // args[1]
                    raise FailToRun()
                elif p.name == FunctionName.MOD:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        if args[1] == 0:
                            raise FailToRun()
                        return args[0] % args[1]
                    raise FailToRun()
                elif p.name == FunctionName.NOT:
                    if isinstance(args[0], bool):
                        return not args[0]
                    raise FailToRun()
                elif p.name == FunctionName.AND:
                    if isinstance(args[0], bool) and isinstance(args[1], bool):
                        return args[0] and args[1]
                    raise FailToRun()
                elif p.name == FunctionName.OR:
                    if isinstance(args[0], bool) and isinstance(args[1], bool):
                        return args[0] or args[1]
                    raise FailToRun()
                elif p.name == FunctionName.EQ:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        return args[0] == args[1]
                    raise FailToRun()
                elif p.name == FunctionName.NE:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        return args[0] != args[1]
                    raise FailToRun()
                elif p.name == FunctionName.LT:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        return args[0] < args[1]
                    raise FailToRun()
                elif p.name == FunctionName.LE:
                    if isinstance(args[0], int) and isinstance(args[1], int):
                        return args[0] <= args[1]
                    raise FailToRun()
                elif p.name == FunctionName.WHERE:
                    if not isinstance(args[0], bool):
                        raise FailToRun()
                    if not isinstance(args[1], int):
                        raise FailToRun()
                    if not isinstance(args[2], int):
                        raise FailToRun()
                    if args[0]:
                        return args[1]
                    else:
                        return args[2]
                else:
                    raise FailToRun()
            else:
                raise AssertionError(f"Invalid program type: {p}")

        try:
            return _run(p)
        except FailToRun:
            return None


@dataclass
class Example:
    inputs: List[Union[int, bool]]
    output: Union[int, bool]


@dataclass
class Sample:
    program: Program
    examples: List[Example]


class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rng: np.random.RandomState,
        max_sample: int,
        max_input: int,
        max_int: int,
        max_depth: int
    ):
        self.rng = rng
        self.max_sample = max_sample
        self.max_input = max_input
        self.max_int = max_int
        self.max_depth = max_depth
        self.interpreter = Interpreter()

    @property
    def tokens(self):
        names = list(FunctionName.__members__.keys())
        m = max(self.max_input, self.max_int)
        return ["input", "(", ")", "True", "False"] + names + list(range(0, m + 1))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        if worker_info is not None:
            worker_id = worker_info.id
        seed = self.rng.randint(0, (2 << 31) - 1) + worker_id
        rng = np.random.RandomState(seed)

        class InternalIterator:
            def __init__(self, parent):
                self.parent = parent
                self.obj_prob = [float(n) for n in range(1, self.parent.max_depth + 1)]
                self.obj_prob = [p / sum(self.obj_prob) - 1e-5 for p in self.obj_prob]

            def _gen_input(self, p: Program) -> List[int]:
                x: List[int] = rng.randint(
                    -(self.parent.max_int + 1), self.parent.max_int + 1, p.n_input
                ).tolist()
                return x

            def _gen_program(self, max_depth: int) -> Program:
                assert max_depth != 0
                if max_depth == 1:
                    cands = ["Input", "Number", "Boolean"]
                else:
                    cands = ["Function"]
                x = rng.choice(cands)
                if x == "Input":
                    return Input(rng.randint(0, self.parent.max_input + 1))
                elif x == "Number":
                    return Number(rng.randint(0, self.parent.max_int + 1))
                elif x == "Boolean":
                    return Boolean(rng.choice([True, False]))
                else:
                    names = list(FunctionName.__members__.values())
                    name = rng.choice(names)
                    arity = FunctionName.arity(name)
                    args = []
                    i_max_depth = rng.randint(0, arity)
                    for i in range(arity):
                        if i == i_max_depth:
                            depth = max_depth - 1
                        else:
                            depth = rng.randint(1, max_depth)
                        args.append(self._gen_program(depth))
                    return Function(name, args)

            def __next__(self) -> Sample:
                depth = rng.multinomial(1, self.obj_prob).nonzero()[0] + 1
                while True:
                    p = self._gen_program(depth)
                    n_sample = rng.randint(1, self.parent.max_sample + 1)
                    inputs = [self._gen_input(p) for _ in range(n_sample)]
                    outs = [self.parent.interpreter.run(p, input) for input in inputs]

                    def cond(x):
                        if x is None:
                            return True
                        else:
                            return abs(x) <= self.parent.max_int

                    if any([cond(out) for out in outs]):
                        continue
                    samples = [Example(input, out) for input, out in zip(inputs, outs)]
                    return Sample(p, samples)

        return InternalIterator(self)
