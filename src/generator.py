"""
generator.py
Core module: Uses LangChain + Ollama to generate test cases
from user stories, feature descriptions, bug reports, or API specs.
"""

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

PLAIN_TEXT_TEMPLATE = """
You are an expert QA Engineer with 10 years of experience writing test cases.

Given the following {input_type}, generate exactly {num_cases} detailed test cases.

Each test case must follow this structure:
TC-001: <Title>
  Preconditions: <what must be true before test>
  Steps:
    1. <step>
    2. <step>
  Expected Result: <what should happen>
  Priority: High / Medium / Low
  Type: Positive / Negative / Edge Case

{context_block}

{input_type}:
{user_input}

Generate {num_cases} test cases now:
"""

GHERKIN_TEMPLATE = """
You are an expert QA Engineer. Generate exactly {num_cases} BDD test scenarios
in Gherkin format (Feature, Scenario, Given, When, Then, And).

{context_block}

{input_type}:
{user_input}

Output only valid Gherkin syntax. Start each scenario on a new line.
"""

TABLE_TEMPLATE = """
You are an expert QA Engineer. Generate exactly {num_cases} test cases
in a structured table format using this exact layout:

| TC ID | Title | Precondition | Steps | Expected Result | Priority | Type |
|-------|-------|--------------|-------|-----------------|----------|------|
| TC-001 | ... | ... | ... | ... | High | Positive |

{context_block}

{input_type}:
{user_input}

Generate the table now:
"""

FORMAT_TEMPLATE_MAP = {
    "Plain Text": PLAIN_TEXT_TEMPLATE,
    "Gherkin (BDD)": GHERKIN_TEMPLATE,
    "Table Format": TABLE_TEMPLATE
}


# ─────────────────────────────────────────────
# TestCaseGenerator Class
# ─────────────────────────────────────────────

class TestCaseGenerator:
    """
    Generates test cases using Ollama LLM via LangChain.

    Args:
        model_name (str): Ollama model to use. Default is 'llama3'.
    """

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name, temperature=0.3)

    def generate(
        self,
        user_input: str,
        input_type: str = "User Story",
        output_format: str = "Plain Text",
        num_cases: int = 5,
        context: str = ""
    ) -> str:
        """
        Generate test cases from the given input.

        Args:
            user_input   : The feature/story/bug text
            input_type   : Type label shown in prompt
            output_format: Plain Text / Gherkin (BDD) / Table Format
            num_cases    : How many test cases to generate
            context      : Optional RAG context from ChromaDB

        Returns:
            str: Generated test cases as a string
        """
        template_str = FORMAT_TEMPLATE_MAP.get(output_format, PLAIN_TEXT_TEMPLATE)

        context_block = ""
        if context:
            context_block = f"Use the following reference documentation to guide test case generation:\n\"\"\"\n{context}\n\"\"\""

        prompt = PromptTemplate(
            input_variables=["input_type", "num_cases", "user_input", "context_block"],
            template=template_str
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        result = chain.invoke({
            "input_type": input_type,
            "num_cases": num_cases,
            "user_input": user_input,
            "context_block": context_block
        })

        # LangChain returns dict; extract text
        if isinstance(result, dict):
            return result.get("text", str(result))
        return str(result)


# ─────────────────────────────────────────────
# CLI usage (for testing without Streamlit)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample_story = """
    As a user, I want to log in to the application using my email and password
    so that I can access my dashboard securely.
    Acceptance Criteria:
    - Valid credentials should redirect to dashboard
    - Invalid credentials should show error message
    - Account should lock after 5 failed attempts
    """

    gen = TestCaseGenerator(model_name="llama3")
    output = gen.generate(
        user_input=sample_story,
        input_type="User Story",
        output_format="Plain Text",
        num_cases=5
    )
    print(output)
