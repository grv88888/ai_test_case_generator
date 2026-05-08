"""
cli.py
Command-line interface for the AI Test Case Generator.
Use this to generate test cases directly from terminal without Streamlit.

Usage:
    python cli.py --input "your user story here" --format plain --count 5
    python cli.py --file sample_inputs/login_story.txt --format gherkin --count 3
"""

import argparse
from src.generator import TestCaseGenerator


def main():
    parser = argparse.ArgumentParser(
        description="🧪 AI Test Case Generator CLI (Ollama + LangChain)"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="User story or feature description as a string"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a .txt file containing the input"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["plain", "gherkin", "table"],
        default="plain",
        help="Output format: plain | gherkin | table (default: plain)"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=5,
        help="Number of test cases to generate (default: 5)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3",
        help="Ollama model to use: llama3 | mistral | gemma (default: llama3)"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="User Story",
        choices=["User Story", "Feature Description", "Bug Report", "API Endpoint"],
        help="Input type label (default: User Story)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Optional: save output to a file path"
    )

    args = parser.parse_args()

    # Resolve input
    user_input = ""
    if args.input:
        user_input = args.input
    elif args.file:
        with open(args.file, "r") as f:
            user_input = f.read()
    else:
        print("❌ Please provide --input or --file")
        parser.print_help()
        return

    # Map format arg to display string
    format_map = {
        "plain": "Plain Text",
        "gherkin": "Gherkin (BDD)",
        "table": "Table Format"
    }
    output_format = format_map[args.format]

    print(f"\n🚀 Generating {args.count} test cases using {args.model}...")
    print(f"   Format  : {output_format}")
    print(f"   Type    : {args.type}")
    print("-" * 60)

    generator = TestCaseGenerator(model_name=args.model)
    result = generator.generate(
        user_input=user_input,
        input_type=args.type,
        output_format=output_format,
        num_cases=args.count
    )

    print(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"\n✅ Output saved to: {args.output}")


if __name__ == "__main__":
    main()
