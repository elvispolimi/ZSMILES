from pathlib import Path
from argparse import ArgumentParser
import subprocess
import os
import filecmp

if __name__ == "__main__":
    # parse the program options
    parser = ArgumentParser(description="Tests")
    parser.add_argument("--test_data", type=Path, help="folder with data for tests")
    parser.add_argument(
        "--executable", type=str, help="path to executable to be tested"
    )
    parser.add_argument(
        "--compress", default=False, action="store_true", help="test compress"
    )
    parser.add_argument(
        "--decompress", default=False, action="store_true", help="test decompress"
    )
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable CUDA implementation"
    )
    parser.add_argument(
        "--hip", default=False, action="store_true", help="Enable HIP implementation"
    )
    args = parser.parse_args()

    opts = []
    output = Path(f"{args.test_data}")
    reference = Path(f"{args.test_data}")
    if args.compress:
        reference /= "drugs.zsmi"
        output /= "test.zsmi"
        opts += ["-c"]
        opts += ["-i"]
        opts += [str(args.test_data / "drugs.smi")]
        opts += ["-o"]
        opts += [str(output)]
    elif args.decompress:
        reference /= "drugs.smi"
        output /= "test.smi"
        opts += ["-d"]
        opts += ["-i"]
        opts += [str(args.test_data / "drugs.zsmi")]
        opts += ["-o"]
        opts += [str(output)]
    else:
        raise RuntimeError("You should either use --compress or --decompress")

    if args.cuda:
        opts += ["--cuda"]
    if args.hip:
        opts += ["--hip"]

    # execute the test and make sure that it has executed properly
    rc = subprocess.run(
        [f"{args.executable}"] + opts,
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard error
        text=True,  # Decode output as text
    )
    if rc.returncode != 0:
        raise RuntimeError(
            f'Test "{" ".join([f"{args.executable}"] + opts)}" have failed with return code {rc.returncode}\n{rc.stderr}'
        )

    # Check if equal
    if not filecmp.cmp(reference, output):
        raise RuntimeError("File are not equal")

    if output.exists():
        os.remove(output)

    print("Test passed")
