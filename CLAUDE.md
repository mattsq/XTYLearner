- When generating or modifying YAML files:
1. ALWAYS validate YAML syntax before saving
2. Use 2 spaces for indentation (never tabs)
3. For GitHub Actions: verify job/step structure, required fields, and action versions
4. Common pitfalls to check:
   - Proper quoting for special characters
   - Correct list syntax (- vs no dash)
   - Valid GitHub Actions context syntax (${{ }})
- Always use uv to install packages and run python scripts.