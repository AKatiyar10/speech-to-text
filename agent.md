# AGENT GUIDELINES & PROJECT STANDARDS

**Crucial**: You must read and follow this file at the start of every session.

## 1. Project Structure
Maintain the following directory structure strictly:
```
/
├── backend/                # Python/FastAPI backend
│   ├── tests/              # [NEW] All backend tests (pytest)
│   ├── venv/               # Virtual environment
│   ├── *.py                # Source files
│   └── requirements.txt
├── frontend/               # Frontend code
└── agent.md                # This file
```

## 2. Test Driven Development (TDD)
**MANDATORY**: You must strictly follow TDD.
1.  **Red**: Write a failing test for the new feature or bug fix in `backend/tests/` BEFORE writing any implementation code.
2.  **Green**: Write the minimum code necessary to pass the test.
3.  **Refactor**: Improve the code quality while keeping tests green.

*   **Tools**: Use `pytest` for backend tests.
*   **Location**: All tests must be in `backend/tests/` mirroring the module structure (e.g., `backend/tests/test_refinement_engine.py`).

## 3. Production Quality Standards
*   **Logging**: Use the standard `logging` module. Do not use `print` statements.
*   **Error Handling**: Use `try/except` blocks with specific exceptions. Fail gracefully and log errors.
*   **Configuration**: No hardcoded credentials or API keys. Use environment variables or configuration files.
*   **Type Hinting**: All Python code must be fully type-hinted.
*   **Documentation**: All classes and major functions must have docstrings.

## 4. Workflow Check
Before finishing a task:
1.  Did I write a test?
2.  Did the test pass?
3.  Is the code linted/clean?
4.  Did I update `requirements.txt` if needed?
