import os

import yaml

from app.datasets.toy import Interpreter, Parser


def test_eval_file():
    eval_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "toy", "eval.yaml"
    )

    with open(eval_file) as file:
        data = yaml.load(file)

    parser = Parser()
    interpreter = Interpreter()
    for value in data:
        program = parser.parse(value["program"])
        assert program is not None
        for example in value["examples"]:
            out = interpreter.run(program, example["input"])
            assert out == example["output"]
