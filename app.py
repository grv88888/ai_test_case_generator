import streamlit as st
from src.generator import TestCaseGenerator
from src.rag_handler import RAGHandler
from src.utils import export_to_csv, export_to_json
import json

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Test Case Generator",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 AI-Powered Test Case Generator")
st.markdown("**Powered by Ollama (LLaMA 3) + LangChain + ChromaDB**")
st.divider()

# ─────────────────────────────────────────────
# Sidebar Config
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Select LLM Model",
        ["llama3", "mistral", "gemma"],
        index=0
    )

    output_format = st.radio(
        "Output Format",
        ["Plain Text", "Gherkin (BDD)", "Table Format"],
        index=0
    )

    use_rag = st.toggle("Enable RAG (use uploaded docs)", value=False)

    st.divider()
    st.markdown("### 📁 Upload Reference Docs (for RAG)")
    uploaded_file = st.file_uploader(
        "Upload .txt or .pdf test spec",
        type=["txt", "pdf"],
        disabled=not use_rag
    )

    if uploaded_file and use_rag:
        with st.spinner("Indexing document into ChromaDB..."):
            rag = RAGHandler()
            rag.index_document(uploaded_file)
        st.success("✅ Document indexed!")

    st.divider()
    st.markdown("**Made by Gaurav Rathod**")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/gauravrathod-37251b148)")


# ─────────────────────────────────────────────
# Main Input Area
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")

    input_type = st.selectbox(
        "Input Type",
        ["User Story", "Feature Description", "Bug Report", "API Endpoint"]
    )

    user_input = st.text_area(
        f"Enter your {input_type}",
        height=200,
        placeholder=f"Paste your {input_type.lower()} here..."
    )

    num_cases = st.slider("Number of Test Cases", min_value=3, max_value=15, value=5)

    generate_btn = st.button("🚀 Generate Test Cases", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# Output Area
# ─────────────────────────────────────────────
with col2:
    st.subheader("📋 Generated Test Cases")

    if generate_btn:
        if not user_input.strip():
            st.error("⚠️ Please enter some input before generating.")
        else:
            with st.spinner(f"Generating {num_cases} test cases using {model_name}..."):
                try:
                    generator = TestCaseGenerator(model_name=model_name)

                    context = ""
                    if use_rag and uploaded_file:
                        rag = RAGHandler()
                        context = rag.retrieve_context(user_input)

                    result = generator.generate(
                        user_input=user_input,
                        input_type=input_type,
                        output_format=output_format,
                        num_cases=num_cases,
                        context=context
                    )

                    st.session_state["last_result"] = result
                    st.session_state["last_input"] = user_input

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure Ollama is running: `ollama serve`")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.text_area("Output", value=result, height=350)

        st.divider()
        st.markdown("### 💾 Export")
        ecol1, ecol2 = st.columns(2)

        with ecol1:
            csv_data = export_to_csv(result)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name="test_cases.csv",
                mime="text/csv",
                use_container_width=True
            )

        with ecol2:
            json_data = export_to_json(result, st.session_state.get("last_input", ""))
            st.download_button(
                "⬇️ Download JSON",
                data=json_data,
                file_name="test_cases.json",
                mime="application/json",
                use_container_width=True
            )

# ─────────────────────────────────────────────
# History Section
# ─────────────────────────────────────────────
st.divider()
with st.expander("📜 Session History"):
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if generate_btn and "last_result" in st.session_state:
        st.session_state["history"].append({
            "input": st.session_state.get("last_input", ""),
            "output": st.session_state["last_result"]
        })

    if st.session_state.get("history"):
        for i, item in enumerate(reversed(st.session_state["history"]), 1):
            st.markdown(f"**#{i} Input:** {item['input'][:80]}...")
            st.code(item["output"][:300] + "...", language="text")
    else:
        st.info("No history yet. Generate some test cases!")
