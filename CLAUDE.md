## Code Style Guidelines
- Write simple, clean, and readable code with minimal indirection
- Avoid unnecessary object attributes, local variables, and config variables
- Avoid tight coupling between files and modules.
- Avoid object/struct attribute redirections
- No redundant abstractions or duplicate code
- Each function should do one thing well
- Use clear, descriptive names in camel case
- Public methods MUST have full documentation
- Once you finish the task, DO NOT write documentations unless the task specifically requires it.
- You MUST check and test the code you have written except for formatting/typing changes

## Testing Instructions
- Run lint and typecheckers and fix any lint and typecheck errors
- Carefully read the code, find and fix redundancies, duplications,
 inconsistencies, errors, and AI slop in the code
- Generate comprehensive tests so that you achieve 100% branch coverage
- Tests MUST NOT use mocks, patches, or any form of test doubles
- Integration tests are HIGHLY encouraged
- You MUST not add tests that are redundant or duplicate of existing
 tests or does not add new coverage over existing tests
- Generate meaningful stress tests for the code if you are
 optimizing the code for performance
- Each test should be independent and verify actual behavior

## Use web tools when you need to:
- Look up API documentation or library usage from the internet
- Find examples of similar implementations
- Understand existing code in the project
- Augment recent knowledge and to perform web based tasks
- Read papers from the internet to understand concepts and algorithms

## After you have implemented the task, aggresively and carefully simplify and clean up the code
 - Remove unnecessary conditional checks
 - Make sure that the code is still working correctly
 - Simplify and clean up the test code
