# EvalQAG-A-Framework-for-Automatic-Complex-QA-Generation 🌱

**EvalQAG** is a modular and scalable framework for **automatically generating**, **evaluating**, and **filtering high-quality question–answer (QA) pairs** from complex renewable energy policy documents. It is designed to address the challenges posed by the **dense, legal, and domain-specific language** found in such documents—especially those related to clean energy incentives, tax credits, and solar programs.

The framework leverages **large language models (LLMs)** in a multi-stage pipeline to support scalable construction of diverse QA datasets. It is optimized for five distinct question types:

- ✔️ Yes/No  
- ⚖️ Yes/No with condition  
- 📜 Legal Obligation  
- 📚 Factual  
- 📝 Descriptive

---

## 🎯 Why EvalQAG?

Traditional QA datasets are often poorly suited to the complex and formal structure of policy documents. **EvalQAG** bridges this gap by introducing:

- **Multi-Stage Generation Pipeline**: LLM-driven, role-aware prompt design that considers document structure and metadata (e.g., title, sector, incentive amounts).
- **Fine-Grained Evaluation Dimensions**: LLM-based scoring of generated QA pairs across:
  - **Accuracy**
  - **Completeness**
  - **Groundedness**
  - **Relevance**
  - **Intent**
- **Local and Global Filtering**: Combines within-document QA selection with dataset-wide semantic filtering using sentence embeddings.
- **Policy-Oriented Design**: Focused on aiding users in understanding incentive eligibility, legal compliance, financial benefits, and administrative procedures.

---

## 📦 Dataset: RePolicyQA [Link 🔗](https://www.dropbox.com/scl/fo/0fkernnzvxweokjs6w73m/ALdEN5KHBSH7eERUTmBoH5M?rlkey=dnr4xxkbqsbadr6jea9ok10r0&st=hxchxdzv&dl=0)

As part of this framework, we introduce **RePolicyQA** (Renewable Energy Policy QA Dataset), a high-quality, LLM-evaluated QA dataset specifically curated from official U.S. solar and energy incentive programs.

Key properties:

- ✅ Thousands of QA pairs spanning factual, legal, and descriptive questions
- 📄 Derived from real-world policy documents sourced from databases like DSIRE
- 🧪 All QA pairs evaluated across 5 dimensions using structured LLM prompting
- 🧹 Includes filtered subsets via local and global filtering stages

This dataset can be used for training and evaluating QA models for **legal, policy, and sustainability domains**.

---

## ✨ Key Contributions

- 📌 First scalable QA generation pipeline tailored for energy policy understanding
- 🧠 Evaluation prompts that incorporate metadata (e.g., user sector, program type)
- 🔍 Dataset-level and per-document QA quality control using LLMs
- 💬 Supports downstream use cases in **regulatory compliance**, **public benefit communication**, and **automated document summarization**

---

## ℹ️ Disclaimer

- 📂 RePolicyQA includes responses generated using large language models (LLMs). Although the models were constrained to follow the content of the original documents, the generated answers may differ from the exact text, and may contain factual inconsistencies or unintended biases inherent to LLMs. All data sources used were publicly available, and care was taken to ensure compliance with licensing and usage restrictions. Users are advised to verify LLM-generated content before downstream use.



## 👇 Continue reading below for module structure, pipeline stages, and shell commands to run the system.
Coming Soon...
