# ✅ Pull Request (PR) Guidelines

Follow these guidelines to maintain clarity, quality, and consistency across contributions.

---

## 📏 PR Size & Scope

* Keep PRs **under 500 lines of changes** (additions + deletions).
* Submit **one logical change per PR**.
* **Avoid mixing** refactoring, new features, and bug fixes in the same PR.
* For large features, **split into smaller, reviewable PRs**.

---

## 📜 PR Description

Your PR description should answer ***what***, ***why***, and ***how***:

* Provide a **clear summary** of the changes.
* Include **screenshots or demos** for any UI changes.
* Mention any **breaking changes**.
* Add **testing instructions** (manual steps or test scripts).
* Link related issues or tickets: `Fixes #123`, `Closes #456`.

Use the appropriate PR template:

* 🐞 Bug Fix
* ✨ New Feature
* 📚 Documentation Update
* 🔐 Security Update
* ♻️ Code Refactoring
* 🚀 Performance Improvement
* 🧪 Test Update

---

## 🔍 Review Process

* PR requires **at least one approval** before merge.
* Respond to **all reviewer comments** constructively.
* **Update the PR description** if scope or implementation changes.
* **Resolve merge conflicts** before requesting review.
* Tag relevant team members or domain experts using `@username`.

---

## 🚦 Before Merging

Ensure the following are completed:

* ✅ All CI checks pass (tests, linters, build, etc.)
* 📚 Documentation is updated if applicable
* 🔐 No secrets, tokens, or sensitive data exposed
* 🧹 Commits are **squashed** or organized meaningfully
* 🌿 Delete the feature branch after merge (if not needed)

---

## 🏇 PR Title Format

Use [**Conventional Commits**](https://www.conventionalcommits.org/en/v1.0.0/) for clarity in commit history and automation:

```
<type>(<scope>): <description>
```

### Common Types:

* `feat`: New feature
* `fix`: Bug fix
* `docs`: Documentation only
* `refactor`: Code restructuring (no behavior change)
* `perf`: Performance improvement
* `test`: Adding or updating tests
* `chore`: Maintenance tasks (e.g., build scripts)

### Examples:

* `feat(auth): Add OAuth2 support`
* `fix(api): Handle null response in user endpoint`
* `docs: Update installation guide`
* `refactor(ui): Extract common button component`
* `test(core): Add edge case tests for parser`
