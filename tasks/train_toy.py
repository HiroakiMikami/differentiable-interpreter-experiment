import argparse
import torch
import app
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str)
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--max_value", type=int, default=5)

    args = parser.parse_args()
    interpreter = app.Interpreter(256, 1024, app.toy.arity)
    app.train_extractor(
        interpreter,
        app.toy.functions,
        args.max_value,
        32,
        100000,
        1e-4,
        os.path.join(args.out, "extractor"),
    )
    app.train(
        interpreter,
        app.toy.functions,
        32,
        100000,
        1e-4,
        os.path.join(args.out, "train"),
    )

    torch.save(interpreter, os.path.join(args.out, "interpreter.pt"))
    torch.save(interpreter.state_dict(), os.path.join(args.out, "interpreter_state.pt"))



if __name__ == "__main__":
    main()
