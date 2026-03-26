# Repository Guidelines

## Project Scope
- This repository is a LaTeX thesis/template project.
- Treat `thesis.tex`, `mwe-paper.tex`, `mwe-bachelor.tex`, `thesisexample.tex`, `hsmw-thesis.cls`, `literature.bib`, and files under `assets/` as source files.
- Treat generated LaTeX outputs such as `*.pdf`, `*.aux`, `*.bcf`, `*.run.xml`, `*.lof`, `*.lot`, `*.toc`, `*.gl*`, `*.sl*`, `*.ac*`, `*.idx`, `*.ind`, `*.ilg`, `*.ist`, `*.alg`, and similar build artifacts as derived files unless the user explicitly asks to edit them.

## Working Rules
- Prefer minimal, source-focused changes.
- Do not edit generated build artifacts unless the user explicitly requests it.
- When investigating rendering or compilation issues, inspect the corresponding `.tex`, `.cls`, `.bib`, and asset files first.
- Preserve the existing LaTeX style and macro conventions used in the repository.
- If you add new assets, keep them under `assets/`.

## Validation
- If validation is needed, prefer targeted LaTeX compilation commands for the affected document.
- Do not treat changes in generated files alone as source changes.

## Git Hygiene
- Keep `.gitignore` aligned with common LaTeX build outputs so diffs stay focused on source files.
