import argparse
import torch
import app
import os
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--max_value", type=int, default=5)

    args = parser.parse_args()
    interpreter: app.Interpreter = torch.load(args.model)

    with open(os.path.join("datasets", "toy", "eval.yaml")) as file:
        dataset = yaml.load(file)

    constants = list(range(-args.max_value, args.max_value + 1))

    summary = {}
    for data in dataset:
        name = data["name"]
        examples = data["examples"]
        n_input = len(examples[0]["input"])
        print(name)

        program = app.create_program(
            args.length,
            n_input + len(constants),
            app.toy.arity,
            interpreter.z_dim,
        )
        app.infer(
            program,
            interpreter,
            [constants + example["input"] for example in examples],
            [example["output"] for example in examples],
            10000,
            os.path.join(args.out, name),
            lr=1e-4
        )

        decoded = program.decode(interpreter, len(app.toy.functions))
        torch.save(decoded, os.path.join(args.out, name, "decoded.pt"))

        testcases = []
        for example in examples:
            i = example["input"]
            env = constants + i
            for fid, arg_id in zip(decoded.func_ids, decoded.arg_idx):
                _args = []
                for id in arg_id:
                    _args.append(env[id])
                out = app.toy.functions[fid](_args)
                env.append(out)
            testcases.append({
                "input": i,
                "output": example["output"],
                "actual": out
            })
        summary[name] = {
            "func_ids": decoded.func_ids,
            "arg_idx": decoded.arg_idx,
            "testcases": testcases,
        }
        print(summary[name])

    print(summary)
    torch.save(summary, os.path.join(args.out, "summary.pt"))


if __name__ == "__main__":
    main()
