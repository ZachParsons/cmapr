# Claude Code Rules for concept-mapper

## Project Context
This is an NLP project for concept mapping and linguistic analysis using spaCy and Stanza.

## Code Style & Quality

### Python Standards
- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and single-purpose
- Use descriptive variable names

### Functional Programming Style
- **Prefer expressions over statements** - Use functional constructs when possible
- **Prefer functional syntax over OOP** - Functions and composition over classes when appropriate
- **Design for composability** - Functions should be easy to chain and combine
- Example: Prefer `map()`, list comprehensions, and function composition over imperative loops

### Formatting & Linting
- Use `make format` to run code formatting
- Run `make test` before committing changes
- Check linting with ruff before submitting code

## Testing Requirements

### Test Coverage
- **ALWAYS write tests for new or changed functionality** - No exceptions
- Tests live in `tests/` directory
- Use pytest for all testing
- Run tests with `make test` or `pytest tests/ -v`

### Test Style
- Use descriptive test names that explain what's being tested
- Include both positive and negative test cases
- Test edge cases and error conditions

### Test Data
- **NEVER add redundant test or sample data without approval**
- We have the Philosopher source text - reuse existing test data
- Don't create new sample files for every feature or test
- Ask first before adding new test corpora

## Documentation

### Code Documentation
- Add docstrings to public functions and classes
- Explain WHY not WHAT (code shows what, docs explain why)
- Update docs/ when adding new features

### Documentation Structure
- **NEVER duplicate documentation** - READMEs, usage guides, API references should be centrally located
- Keep documentation in one authoritative place (avoid copies across files)
- Link to central docs rather than duplicating content

### Acronyms & Terminology
- **Always explain acronyms in a centralized place** - Use the README or a glossary
- Define technical terms on first use
- Maintain consistency in terminology across all docs

### Examples & Shell Usage
- **Prefer one-liners over multi-line examples** - Users paste into shell, so make it easy
- **Provide runner functions** - Compose functionality into single-paste functions to avoid multiple paste steps
- Add usage examples to `examples/` for new features
- Keep examples simple and focused
- Test that examples actually work

## Development Workflow

### Dependencies
- This project uses `uv` for package management
- Add dependencies via `uv add <package>`
- Do NOT use pip directly
- **Use modern tooling** - Prefer existing packages over adding new ones
- **Justify overlapping functionality** - If proposing a new package that overlaps with existing tools, explain and justify the change
- Examples: uv vs poetry (uv is faster, modern), NLTK vs Stanza (NLTK for basic NLP, Stanza for advanced parsing)

### Pre-Commit Checks
- **ALWAYS run `make check` before committing** - This installs deps, formats, lints, and tests
- Never commit without running the full check
- Fix all test failures and lint errors before committing
- **Check if documentation needs updating** - Before committing, verify:
  - README.md reflects any new features or changes
  - API reference (docs/api-reference.md) is current
  - Roadmaps or project status docs reflect completion/changes
  - No stale information contradicts your changes

### Git Practices
- Write clear, descriptive commit messages
- Focus commits on single logical changes
- Don't commit generated files or cache directories

## AI Assistant Guidelines

### Communication & Collaboration
- **Ask questions during development** - Don't assume, ask the lead developer for clarification
- **Get feedback early and often** - Check assumptions before implementing
- **Ask before adding GenAI artifacts** - Never create markdown files (tasks, validation docs, etc.) without approval
- **Explain your reasoning** - When suggesting changes or additions, explain the rationale

### AI-Generated Artifacts Storage
- **ALL AI-generated planning artifacts MUST go in the roadmap** - Never create separate task files
- **Consolidate in docs/concept-mapper-roadmap.md** - All todos, completed work summaries, maintenance tasks, future plans
- **Use roadmap sections:**
  - "Ongoing Maintenance Tasks" - Current maintenance todos
  - "Recent Completions" - Recently finished work
  - "Next Steps (Optional Future Work)" - Future enhancements
- **Delete standalone task files** - No TASKS.md, TODO.md, BACKLOG.md, etc.
- **Keep history** - Don't remove completed items, mark them with [x]

### When Writing Code
- Read existing code first to understand patterns
- Match the existing code style and structure
- Don't over-engineer solutions
- Prefer editing existing files over creating new ones

### When Adding Features
- Check if similar functionality already exists
- Follow established patterns in the codebase
- Update tests and docs along with code changes
- **Ask questions before major decisions** - Consult the user on design choices

### What to Avoid
- Don't add unnecessary abstractions
- Don't create helpers for one-off operations
- Don't add comments explaining obvious code
- Don't add error handling for impossible scenarios
- Don't create new markdown documentation files without asking first

## Project-Specific Notes

### NLP Processing
- The project uses NLTK and Stanza (not spaCy currently)
- Lemmatization is preferred for concept matching
- Dependency parsing is used for sentence structure analysis

### Output Conventions
- **NEVER output user-defined functionality to project root**
- **ALWAYS use subdirectories of `./output/`** for generated files
- Structure: `output/corpus/`, `output/terms/`, `output/graphs/`, `output/exports/`
- Keep the project root clean

### CLI Tool
- Main CLI is `concept-mapper` (defined in `src/concept_mapper/cli.py`)
- Test CLI changes with the examples in `examples/`
