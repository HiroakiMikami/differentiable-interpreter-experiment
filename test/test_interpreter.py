from app import Interpreter


def test_interpreter() -> None:
    interpreter = Interpreter(1, 16, 2)
    assert len(list(interpreter.parameters())) == 20
