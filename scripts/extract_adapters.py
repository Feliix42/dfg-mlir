import argparse
import re
import sys

def convertTypeString(typeString):
    typeString = typeString.strip()
    if typeString.startswith("ptr") or typeString.endswith("ptr"):
        return "!llvm.ptr"
    if typeString.startswith("void"):
        return "()"
    raise ValueError("Unknown type " + typeString)

def main():
    parser = argparse.ArgumentParser(description="Extract dpm adapter function declarations from LLVMIR code")
    parser.add_argument("file", type=str, help="The path to the LLVMIR file")
    args = parser.parse_args()

    with open(args.file, 'r') as file:
        content = file.read()

        #matches = re.findall(r"^.*declare (.*) (@.*dpm_adapter)\((.*)\)", content, re.MULTILINE)
        matches = re.findall(r"^.*define (.*) (@.*dpm_adapter)\((.*)\)", content, re.MULTILINE)

        for returnType, funcName, argumentTypes in matches:
            try:
                arguments = ", ".join([convertTypeString(_e) for _e in argumentTypes.split(",")])
                print(f"llvm.func {funcName}({arguments}) -> {convertTypeString(returnType)}")
            except ValueError as ex:
                print(f"{ex} for {funcName}", file=sys.stderr)

if __name__ == "__main__":
    main()
