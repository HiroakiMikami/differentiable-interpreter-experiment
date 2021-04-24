import numpy as np
import torch

from app.datasets.toy import (
    Boolean,
    Function,
    FunctionName,
    Input,
    Interpreter,
    Number,
    Parser,
    RandomDataset,
)


def test_parse():
    parser = Parser()
    assert parser.parse("input 0") == Input(0)
    assert parser.parse("True") == Boolean(True)
    assert parser.parse("False") == Boolean(False)
    assert parser.parse("10") == Number(10)
    assert parser.parse("ID ( 10 )") == Function(FunctionName.ID, [Number(10)])


def test_parse_invalid_program():
    parser = Parser()
    assert parser.parse("input") is None
    assert parser.parse("10.0") is None
    assert parser.parse("foo") is None
    assert parser.parse("ID ( 10") is None
    assert parser.parse("ID ( ADD ( 20 30 )") is None


def test_run():
    interpreter = Interpreter()
    assert interpreter.run(Input(0), [0]) == 0
    assert interpreter.run(Boolean(True), [])
    assert interpreter.run(Number(10), []) == 10
    assert interpreter.run(Function(FunctionName.ID, [Number(10)]), []) == 10
    assert interpreter.run(Function(FunctionName.NEG, [Number(10)]), []) == -10
    assert interpreter.run(Function(FunctionName.ADD, [Number(2), Number(3)]), []) == 5
    assert interpreter.run(Function(FunctionName.SUB, [Number(2), Number(3)]), []) == -1
    assert interpreter.run(Function(FunctionName.MUL, [Number(2), Number(3)]), []) == 6
    assert interpreter.run(Function(FunctionName.DIV, [Number(2), Number(3)]), []) == 0
    assert interpreter.run(Function(FunctionName.MOD, [Number(2), Number(3)]), []) == 2
    assert interpreter.run(Function(FunctionName.NOT, [Boolean(False)]), [])
    assert interpreter.run(
        Function(FunctionName.AND, [Boolean(True), Boolean(True)]), []
    )
    assert interpreter.run(
        Function(FunctionName.OR, [Boolean(True), Boolean(False)]), []
    )
    assert not interpreter.run(Function(FunctionName.EQ, [Number(2), Number(3)]), [])
    assert interpreter.run(Function(FunctionName.NE, [Number(2), Number(3)]), [])
    assert interpreter.run(Function(FunctionName.LT, [Number(2), Number(3)]), [])
    assert interpreter.run(Function(FunctionName.LE, [Number(2), Number(3)]), [])
    assert interpreter.run(
        Function(FunctionName.WHERE, [Boolean(True), Number(10), Number(20)]), []
    ) == 10


def test_run_invalid_program():
    interpreter = Interpreter()
    assert interpreter.run(Input(0), []) is None
    assert interpreter.run(Function(FunctionName.ID, []), []) is None
    assert interpreter.run(Function(FunctionName.NEG, []), []) is None
    assert interpreter.run(Function(FunctionName.ADD, [Number(2)]), []) is None
    assert interpreter.run(Function(FunctionName.SUB, [Number(2)]), []) is None
    assert interpreter.run(Function(FunctionName.MUL, [Number(2)]), []) is None
    assert interpreter.run(
        Function(FunctionName.DIV, [Number(2), Number(0)]), []
    ) is None
    assert interpreter.run(Function(FunctionName.DIV, [Number(2)]), []) is None
    assert interpreter.run(Function(FunctionName.MOD, [Number(2)]), []) is None
    assert interpreter.run(Function(FunctionName.NOT, []), []) is None
    assert interpreter.run(Function(FunctionName.NOT, [Number(10)]), []) is None
    assert interpreter.run(Function(FunctionName.AND, [Boolean(True)]), []) is None
    assert interpreter.run(
        Function(FunctionName.AND, [Boolean(True), Number(10)]), []
    ) is None
    assert interpreter.run(Function(FunctionName.OR, [Boolean(True)]), []) is None
    assert interpreter.run(
        Function(FunctionName.OR, [Boolean(True), Number(10)]), []
    ) is None
    assert interpreter.run(Function(FunctionName.EQ, [Number(2)]), []) is None
    assert interpreter.run(Function(FunctionName.NE, [Number(2)]), []) is None
    assert interpreter.run(Function(FunctionName.LT, [Number(2)]), []) is None
    assert interpreter.run(Function(FunctionName.LE, [Number(2)]), []) is None
    assert interpreter.run(
        Function(FunctionName.WHERE, [Boolean(True), Number(20)]), []
    ) is None
    assert interpreter.run(
        Function(FunctionName.WHERE, [Number(10), Number(10), Number(20)]), []
    ) is None


def test_dataset():
    dataset = RandomDataset(np.random.RandomState(0), 2, 2, 100, 3)
    interpreter = Interpreter()
    for i, sample in enumerate(dataset):
        if i == 100:
            break
        for example in sample.examples:
            assert example.output is not None
            assert interpreter.run(sample.program, example.inputs) == example.output


def test_multiworker_dataset():
    dataset = RandomDataset(np.random.RandomState(0), 2, 2, 100, 3)
    loader = torch.utils.data.DataLoader(
        dataset, 1, collate_fn=lambda x: x, num_workers=2
    )
    samples = []
    for i, sample in enumerate(loader):
        samples.append(sample[0])
        if i == 1:
            break
    assert samples[0].program != samples[1].program
