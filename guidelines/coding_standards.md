# 🧑‍💻 Coding Standards

Adhering to consistent coding standards ensures maintainability, readability, and collaboration across the codebase.

---

## 🐍 Python

* Use **type hints** for all function arguments and return types
* Follow the **PEP 8** style guide
* Limit line length to **100 characters**
* Use **descriptive and meaningful variable names**
* Prefer **f-strings** over `.format()` or `%` formatting for string interpolation

---

## 🔀 Git Commit Standards

* Use **[Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/)** format:
  `type(scope): description`
* Common commit types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
* Keep the **commit message under 72 characters**
* Reference issues when relevant (e.g., `Fixes #123`)

---

## 🧱 Code Organization

* Define **one major class per file** for clarity
* **Group related functions and classes into modules**
* Use `__init__.py` to define the **public API** of a package
* Try to keep **functions under 50 lines** for readability and testability

---

## 🧪 Testing Standards

* All new features **must include tests**
* Maintain **test coverage above 80%**
* Use **pytest** as the primary testing framework
* Always test **edge cases and error conditions**
* **Mock external dependencies** (e.g., APIs, databases) in unit tests

---

## 📚 Documentation

* All **public functions and classes must include docstrings**
* Use **Google-style** docstring format
* Provide **usage examples** for complex or non-obvious functions
* Keep all **README and module-level documentation up to date**
