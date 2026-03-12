# Contributing

Thank you for your interest in contributing to **Autonomous Paper Trader**.

This project is currently maintained primarily by a single developer and is still in an active `0.x` development stage. Contributions are welcome, but they should align with the project scope, coding style, and milestone goals.

## Project Scope

This repository focuses on building an educational and experimental paper trading / backtesting system.

Current priorities include:

- data ingestion and preprocessing
- strategy development
- broker, portfolio, and simulation engine components
- evaluation, explainability, and demo-oriented improvements

Contributions outside the current roadmap may be reviewed more slowly or may not be accepted.

## Before You Start

Please:

1. read the README and existing issues
2. check whether the idea is already planned or in progress
3. open or reference an issue before starting larger changes

For small fixes, documentation improvements, or minor refactors, a direct pull request is usually fine.

## Development Guidelines

Please try to keep contributions:

- focused and scoped to a single purpose
- consistent with the existing project structure
- readable, deterministic, and easy to review
- aligned with the current milestone or issue goals

Avoid unrelated refactors in the same pull request.

## Branching

Please do not work directly on `main`.

Create a feature or fix branch such as:

- `feature/add-order-builder`
- `fix/broker-cash-validation`
- `docs/update-readme`

## Pull Requests

When opening a pull request:

- link the related issue when possible
- clearly explain **what changed** and **why**
- keep the PR as small and reviewable as possible
- update documentation if behavior or usage changes
- make sure the code is in a working state before submitting

A good PR should answer:

- What was added, changed, or fixed?
- Why is this change needed?
- Are there any important notes, assumptions, or limitations?

## Code Style

Please follow the existing style of the repository.

General expectations:

- use clear and descriptive names
- prefer simple and explicit logic
- avoid unnecessary abstractions
- keep functions and modules reasonably focused
- write code that is easy to test and reason about

## Issues

Issues are welcome for:

- bugs
- feature proposals
- documentation improvements
- refactoring suggestions
- roadmap-aligned enhancements

When opening an issue, please be as specific as possible.

Include:

- the problem
- the expected behavior
- the current behavior
- reproduction steps, if applicable
- suggested direction, if relevant

## Discussions and Decision Making

Because this is a solo-maintained project, not every suggestion will be accepted.

Priority is given to changes that:

- support the current roadmap
- improve reliability and maintainability
- keep the system simple and deterministic
- strengthen the educational and demo value of the project

## Security

Please do **not** report security issues through public GitHub issues.

For security-related concerns, follow the instructions in [SECURITY.md](./SECURITY.md).

## License

By contributing to this repository, you agree that your contributions will be licensed under the same license as the project.