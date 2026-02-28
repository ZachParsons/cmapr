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

## Graph Diagram Constraints

These rules apply to all graph construction and visualization code (`graph/`, `export/`, `cli.py` graph/export commands):

1. **No duplicate nodes** — nodes with identical labels must be consolidated into one. All edges from duplicates are re-wired to the canonical node (weights summed, evidence concatenated). Log a `WARNING` for each consolidation so the upstream source can be investigated and fixed. Enforced in `consolidate_duplicate_labels()` (`graph/operations.py`), called automatically in `to_d3_dict()`.

2. **No unconnected nodes** — every node must have at least one edge. Truly isolated nodes (degree 0) trigger a co-occurrence fallback (`connect_isolated_nodes()`); if no co-occurrence partner exists, log an `ERROR`. Any node still isolated at export time is logged as an `ERROR` and dropped from the diagram. Enforced in `find_isolated_nodes()` / `connect_isolated_nodes()` (`graph/operations.py`) and `to_d3_dict()`.

3. **Edge labels must always be text derived from the source text** — never raw numbers. Priority order: verb from SVO metadata → copula → preposition → relation-type fallback string (e.g. `"co-occurs with"`). The numeric weight must never appear as a visible label. Enforced in `to_d3_dict()` (`export/d3.py`) and the HTML template (`export/html.py`).

4. **All edges are directed** — every edge renders as an arrow from source to target, with the arrowhead landing on the target node boundary (not its centre). The edge label reflects the directional relationship as it appears in the source text (subject → verb → object). Enforced in the HTML template arrowhead marker and tick-handler offset logic (`export/html.py`).

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
