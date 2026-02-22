# Claude Code Rules for cmapr

## Project Overview
NLP project for concept mapping and linguistic analysis using NLTK and Stanza.

---

## Code Standards

### Style
- Follow PEP 8 with type hints for all functions
- Prefer functional style: expressions over statements, composition over classes
- Keep functions focused, single-purpose, and composable
- Use descriptive names; avoid unnecessary comments

### Tooling
- **Package manager**: `uv` (not pip)
- **Format**: `ruff format src/ tests/ scripts/`
- **Lint**: `ruff check src/ tests/ scripts/ --fix`
- **Test**: `pytest tests/ -v`
- **Pre-commit**: `make check` (format + lint + test) - **REQUIRED before every commit**

---

## Testing (Non-Negotiable)

### Requirements
- **ALWAYS write tests** for new or changed functionality - no exceptions
- Test edge cases, errors, and both positive/negative scenarios
- Use descriptive test names that explain the behavior
- **When a bug or unexpected result is reported: add a regression test** that asserts the bad behavior cannot recur, before or alongside the fix

### Test Data
- **NEVER add redundant test data** without approval
- Reuse existing samples (eco_spl.txt)
- Ask before adding new test corpora

---

## Documentation

### Principles
- **NEVER duplicate documentation** - one authoritative source per topic
- **Link, don't copy** - reference central docs instead of duplicating
- Explain WHY in docs, not WHAT (code shows what)
- Define acronyms and technical terms on first use

### Structure
- **README.md**: Features, quickstart, project structure
- **docs/api-reference.md**: Detailed API documentation
- **docs/roadmap.md**: Planning, todos, completed work, future plans
- **docs/*.md**: Feature-specific guides (when needed)

### Rules for AI Assistants
**NEVER create new markdown files without explicit approval.** This includes:
- No TASKS.md, TODO.md, BACKLOG.md, VALIDATION.md, etc.
- No duplicate READMEs in subdirectories
- No planning files at project root

**Where to put content:**
- Tasks/todos → `docs/roadmap.md`
- Features → `docs/api-reference.md` or dedicated `docs/<feature>.md`
- Structure → README.md "Project Structure" section

**When in doubt, ASK FIRST.**

### Before Committing
Verify documentation is current:
- README.md reflects new features/changes
- API reference (docs/api-reference.md) is up-to-date
- Roadmap reflects completion status
- No stale information contradicts your changes

---

## Development Workflow

### Dependencies
- Add via `uv add <package>` (never use pip directly)
- Justify new packages if they overlap with existing tools
- Prefer modern, maintained packages

### Git Practices
- Clear, descriptive commit messages
- Focus commits on single logical changes
- Never commit generated files, cache, or virtualenvs
- **NEVER commit without explicit user approval** — always show the diff and wait for the user to confirm they've manually tested and approved before running `git commit`

### AI Assistant Guidelines

**Communication:**
- Ask questions during development - don't assume
- Get feedback early and often
- Explain your reasoning for changes

**When writing code:**
- Read existing code first to understand patterns
- Match existing style and structure
- Prefer editing existing files over creating new ones
- Don't over-engineer or add unnecessary abstractions

**When adding features:**
- Check if similar functionality exists
- Follow established patterns
- Update tests and docs with code changes
- Ask before major design decisions

**What to avoid:**
- Helpers for one-off operations
- Error handling for impossible scenarios
- Comments explaining obvious code
- New markdown files without asking

---

## Project-Specific Rules

### NLP Processing
- Uses NLTK (basic NLP) and Stanza (dependency parsing)
- Lemmatization for concept matching
- Dependency parsing for sentence structure

### Output Files
- **ALWAYS use `./output/` subdirectories** for generated files
- Structure: `output/corpus/`, `output/terms/`, `output/graphs/`, `output/exports/`
- **NEVER output to project root**

### CLI Tool
- Main CLI: `cmapr` (defined in `src/concept_mapper/cli.py`)
- Test CLI changes with examples in `examples/`

### Examples & Shell Usage
- Prefer one-liners for easy copy-paste
- Provide runner functions to avoid multiple paste steps
- Test that examples actually work
