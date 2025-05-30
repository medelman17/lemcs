# Cursor Rules .mdc Template and Best Practices

This guide provides comprehensive templates and best practices for creating effective `.cursor/rules/*.mdc` files based on the latest Cursor documentation and community standards.

## Quick Start Template

```mdc
---
description: Brief description of what this rule does and when it applies
globs: ["**/*.ts", "**/*.tsx"]
alwaysApply: false
---

# Rule Title

Brief explanation of the rule's purpose and when it should be used.

## Guidelines

- Use specific, actionable instructions
- Provide concrete examples
- Reference relevant files when helpful: @filename.ts

## Examples

```typescript
// Good example
const example = "Show what TO do";

// Bad example  
const bad = "Show what NOT to do";
```

## Additional Context

Any additional information, patterns, or references that help the AI understand your requirements.
```

## Rule Types and Frontmatter Options

### Frontmatter Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `description` | string | Brief description shown in UI and used for rule selection | Required for Agent Requested rules |
| `globs` | string/array | Comma-separated file patterns for auto-attachment | Required for Auto Attached rules |
| `alwaysApply` | boolean | If true, rule is always included in context | Optional (default: false) |

### Rule Types Based on Configuration

1. **Always Rules** - `alwaysApply: true`
2. **Auto Attached Rules** - Has `globs` patterns
3. **Agent Requested Rules** - Has `description` but no `globs`
4. **Manual Rules** - Minimal frontmatter, invoked with `@ruleName`

## Complete Templates by Rule Type

### 1. Always Rule Template
```mdc
---
description: Global coding standards applied to all files
alwaysApply: true
---

# Global Coding Standards

These standards apply to all code in this project:

- Use TypeScript for all new code
- Follow ESLint configuration
- Write comprehensive error handling
- Include JSDoc comments for public APIs
```

### 2. Auto Attached Rule Template
```mdc
---
description: React component development guidelines
globs: ["**/*.tsx", "**/*.jsx", "src/components/**/*.ts"]
alwaysApply: false
---

# React Component Guidelines

Automatically applied when working with React components.

## Component Structure
- Use functional components with hooks
- Implement proper prop types with TypeScript
- Follow naming conventions: PascalCase for components

## Styling
- Use Tailwind CSS classes
- Follow responsive design principles
- Reference design system: @src/design-tokens.ts

## Example Component
```tsx
interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary';
  onClick?: () => void;
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'primary', 
  onClick 
}) => {
  return (
    <button 
      className={`btn btn-${variant}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
};
```
```

### 3. Agent Requested Rule Template
```mdc
---
description: Database schema and migration best practices for when working with database-related tasks
---

# Database Guidelines

The AI can choose to apply this rule when database work is detected.

## Schema Design
- Use descriptive table and column names
- Include proper indexes for query performance
- Add foreign key constraints for data integrity

## Migrations
- Always include rollback instructions
- Test migrations on staging before production
- Reference migration template: @db/migration-template.sql

## ORM Best Practices
- Use type-safe queries
- Implement proper connection pooling
- Handle database errors gracefully
```

### 4. Manual Rule Template
```mdc
---
description: API documentation generation standards
---

# API Documentation Standards

Use @api-docs when generating API documentation.

## OpenAPI Specification
- Include detailed descriptions for all endpoints
- Provide example requests and responses
- Document all possible error codes

## Reference Files
@docs/api-spec-template.yaml
@src/types/api.ts
```

## Advanced Templates

### Framework-Specific Rule (Next.js)
```mdc
---
description: Next.js App Router development guidelines and best practices
globs: ["app/**/*.tsx", "app/**/*.ts", "**/*.page.tsx"]
alwaysApply: false
---

# Next.js App Router Guidelines

## File Structure
- Use `page.tsx` for route pages
- Use `layout.tsx` for shared layouts  
- Use `loading.tsx` for loading states
- Use `error.tsx` for error boundaries

## Server vs Client Components
- Default to Server Components
- Mark Client Components with `'use client'` directive
- Use Server Actions for form handling

## Performance
- Implement proper image optimization with `next/image`
- Use dynamic imports for code splitting
- Implement proper caching strategies

## Example Server Component
```tsx
// app/products/page.tsx
import { getProducts } from '@/lib/api';

export default async function ProductsPage() {
  const products = await getProducts();
  
  return (
    <div>
      <h1>Products</h1>
      {products.map(product => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}
```

## Reference Files
@app/layout.tsx
@next.config.js
```

### Testing Rule Template
```mdc
---
description: Testing standards and practices for unit and integration tests
globs: ["**/*.test.ts", "**/*.test.tsx", "**/*.spec.ts", "__tests__/**/*"]
alwaysApply: false
---

# Testing Guidelines

## Test Structure
- Use descriptive test names that explain the scenario
- Follow Arrange-Act-Assert pattern
- Group related tests with `describe` blocks

## Testing Patterns
- Test behavior, not implementation
- Mock external dependencies
- Use data-testid attributes for reliable selectors

## Example Test
```typescript
describe('UserService', () => {
  describe('when creating a user', () => {
    it('should return user with generated ID', async () => {
      // Arrange
      const userData = { name: 'John', email: 'john@test.com' };
      
      // Act
      const result = await userService.create(userData);
      
      // Assert
      expect(result).toMatchObject({
        ...userData,
        id: expect.any(String)
      });
    });
  });
});
```

## Reference Files
@jest.config.js
@__tests__/setup.ts
```

## Best Practices

### Writing Effective Rules

1. **Be Specific and Actionable**
   ```mdc
   # Good
   - Use `const` for variables that don't change
   - Use `let` for variables that need reassignment
   
   # Bad  
   - Use good variable declarations
   ```

2. **Provide Concrete Examples**
   ```mdc
   ## Error Handling
   ```typescript
   try {
     const result = await apiCall();
     return result.data;
   } catch (error) {
     logger.error('API call failed:', error);
     throw new AppError('Failed to fetch data', 500);
   }
   ```
   ```

3. **Reference Relevant Files**
   ```mdc
   Follow the patterns established in:
   @src/utils/api.ts
   @src/types/common.ts
   ```

4. **Keep Rules Focused**
   - Under 500 lines per rule
   - Split large concepts into multiple rules
   - One responsibility per rule

5. **Use Clear Descriptions**
   ```mdc
   # Good description
   description: "React component guidelines for functional components with hooks and TypeScript"
   
   # Bad description  
   description: "React stuff"
   ```

### File Organization

```
.cursor/
└── rules/
    ├── global-standards.mdc          # Always applied
    ├── react-components.mdc          # Auto attached to React files
    ├── api-development.mdc           # Auto attached to API files
    ├── database-guidelines.mdc       # Agent requested for DB work
    ├── testing-standards.mdc         # Auto attached to test files
    └── backend/
        ├── express-routes.mdc        # Backend-specific rules
        └── database-models.mdc
```

### Naming Conventions

- Use kebab-case for filenames: `react-components.mdc`
- Always use `.mdc` extension
- Make names descriptive of the rule's purpose
- Group related rules in subdirectories for large projects

### Common Patterns

#### For Libraries/Frameworks
```mdc
---
description: [Library Name] development guidelines and best practices  
globs: ["**/*.{relevant,extensions}"]
alwaysApply: false
---

# [Library Name] Guidelines

## Installation & Setup
## Core Concepts  
## Best Practices
## Common Patterns
## Examples
## Reference Files
```

#### For Code Quality
```mdc
---
description: Code quality and style guidelines
globs: ["src/**/*.ts", "src/**/*.tsx"]
alwaysApply: false
---

# Code Quality Standards

## Formatting
## Naming Conventions
## Documentation
## Error Handling
## Performance Considerations
```

## Troubleshooting

### Rule Not Being Applied?

1. **Check rule type**: For Agent Requested rules, ensure `description` is defined
2. **Verify file patterns**: For Auto Attached rules, ensure `globs` match your files  
3. **Validate frontmatter**: Ensure YAML syntax is correct
4. **Check file location**: Rules must be in `.cursor/rules/` directory
5. **Review glob syntax**: Use comma-separated patterns without extra quotes

### Frontmatter Gotchas

```mdc
# ✅ Correct
globs: ["**/*.ts", "**/*.tsx"]

# ❌ Incorrect - causes parsing issues
globs: "**/*.ts, **/*.tsx"
```

## Migration from .cursorrules

If migrating from legacy `.cursorrules` files:

1. Create `.cursor/rules/` directory
2. Split content into focused `.mdc` files
3. Add appropriate frontmatter
4. Test rule application
5. Remove legacy `.cursorrules` file

This template provides a solid foundation for creating effective Cursor rules that will improve your AI-assisted development workflow.