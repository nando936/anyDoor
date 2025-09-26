# Subagents Documentation - Claude Code

## Overview

Subagents in Claude Code are specialized AI assistants designed to handle specific types of tasks. They operate with their own context windows, custom system prompts, and configurable tool access, enabling more efficient and focused problem-solving.

## Key Features

- **Separate Context Windows**: Each subagent operates independently, preserving the main conversation context
- **Customized System Prompts**: Tailored instructions for specific expertise areas
- **Configurable Tool Access**: Control which tools each subagent can use
- **Task-Specific Expertise**: Designed for focused, single-purpose operations
- **Proactive Delegation**: Claude Code can automatically delegate tasks to appropriate subagents

## Benefits

1. **Context Preservation**: Prevents pollution of the main conversation
2. **Specialized Domain Expertise**: Each subagent excels at specific tasks
3. **Reusability**: Share subagents across projects and teams
4. **Security**: Limit powerful tools to specific subagents
5. **Scalability**: Multiple subagents can work in parallel

## File Structure

Subagents are defined in Markdown files with YAML frontmatter:

```yaml
---
name: your-subagent-name
description: Description of when this subagent should be invoked
tools: tool1, tool2, tool3  # Optional - inherits all tools if omitted
model: sonnet  # Optional - specify model alias or 'inherit'
---

# Your subagent's system prompt goes here

You are a specialized assistant focused on [specific task].
Your role is to [detailed description of responsibilities].

## Guidelines
- [Specific instruction 1]
- [Specific instruction 2]
- [Specific instruction 3]
```

## Storage Locations

- **Project-level**: `.claude/agents/` (specific to current project)
- **User-level**: `~/.claude/agents/` (available across all projects)

## Creating Subagents

### Using the /agents Command

1. Type `/agents` in Claude Code
2. Select "Create new agent"
3. Provide:
   - Unique name (kebab-case recommended)
   - Clear description of purpose
   - Optional tool restrictions
   - Optional model selection

### Manual Creation

1. Create a new `.md` file in `.claude/agents/`
2. Add YAML frontmatter with configuration
3. Write detailed system prompt below frontmatter
4. Save file with descriptive name

## Configuration Options

### Required Fields
- `name`: Unique identifier for the subagent
- `description`: When and why to use this subagent

### Optional Fields
- `tools`: Comma-separated list of allowed tools (inherits all if omitted)
- `model`: Specific model to use (`inherit`, `sonnet`, `haiku`, etc.)

## Example Subagents

### Code Reviewer
```yaml
---
name: code-reviewer
description: Expert code review specialist for finding bugs and improvements
tools: Read, Grep, Bash
model: inherit
---

You are an expert code reviewer focused on identifying bugs, security issues,
and potential improvements. Analyze code systematically and provide actionable feedback.
```

### Debugger
```yaml
---
name: debugger
description: Specialized in debugging and troubleshooting code issues
tools: Read, Edit, Bash, Grep
model: sonnet
---

You are a debugging specialist. Your role is to identify root causes of bugs,
trace execution paths, and provide fixes with explanations.
```

### Data Scientist
```yaml
---
name: data-scientist
description: Data analysis and machine learning specialist
tools: Read, Write, Bash, WebSearch
model: inherit
---

You are a data science expert specializing in analysis, visualization,
and machine learning implementations.
```

## Best Practices

1. **Start with Claude-Generated Agents**: Ask Claude to create initial versions, then customize
2. **Focus on Single Responsibilities**: Create specialized agents rather than generalists
3. **Write Detailed System Prompts**: Include specific guidelines and examples
4. **Limit Tool Access**: Only grant necessary tools for security
5. **Version Control**: Commit project-specific subagents to your repository
6. **Test Thoroughly**: Validate subagent behavior before production use
7. **Document Usage**: Include clear descriptions of when to use each subagent

## Invocation Methods

### Automatic Delegation
Claude Code automatically delegates when it recognizes tasks matching a subagent's expertise:
- "Review this code for bugs" → code-reviewer subagent
- "Debug this error" → debugger subagent
- "Analyze this dataset" → data-scientist subagent

### Explicit Invocation
Directly request a specific subagent:
- "Use the code-reviewer subagent to check this function"
- "Have the debugger subagent investigate this issue"

## Advanced Usage

### Multi-Agent Workflows
Create complex workflows with multiple specialized subagents:
1. Research subagent gathers information
2. Design subagent creates architecture
3. Implementation subagent writes code
4. Review subagent validates results

### Dynamic Subagent Selection
Claude Code can choose appropriate subagents based on:
- Task complexity
- Required expertise
- Available tools
- Performance requirements

### Chaining Subagents
Subagents can delegate to other subagents for complex tasks:
- Main agent → Research subagent → Data collection subagents
- Lead researcher → Multiple specialized research subagents

## Real-World Applications

### Anthropic's Usage Examples

1. **Ad Generation System**: Two specialized subagents generate hundreds of ads in minutes
2. **Research System**: LeadResearcher creates multiple specialized research subagents
3. **Testing Verification**: Independent subagents verify implementation isn't overfitting
4. **Figma Plugin**: Generates 100+ ad variations programmatically

### Common Use Cases

- **Code Review**: Automated review for pull requests
- **Documentation**: Generate and update documentation
- **Testing**: Create and execute test cases
- **Refactoring**: Identify and implement code improvements
- **Security Audits**: Scan for vulnerabilities
- **Performance Optimization**: Identify bottlenecks

## Troubleshooting

### Common Issues

1. **Subagent Not Found**: Check file location and naming
2. **Tool Access Errors**: Verify tool names in configuration
3. **Context Overflow**: Subagents have separate context limits
4. **Model Conflicts**: Ensure compatible model selection

### Debugging Tips

- Use verbose mode to see subagent selection process
- Check logs for subagent initialization errors
- Validate YAML frontmatter syntax
- Test subagents in isolation before integration

## Future Enhancements

- Dynamic subagent creation during runtime
- Subagent marketplace for sharing
- Performance metrics and analytics
- Enhanced parallel processing capabilities
- Cross-subagent communication protocols

## Resources

- Official Documentation: https://docs.claude.com/en/docs/claude-code/sub-agents
- Best Practices: https://www.anthropic.com/engineering/claude-code-best-practices
- Multi-Agent Systems: https://www.anthropic.com/engineering/multi-agent-research-system
- Building Effective Agents: https://www.anthropic.com/engineering/building-effective-agents

## Summary

Subagents transform Claude Code into a powerful multi-agent system capable of handling complex, specialized tasks efficiently. By creating focused, well-designed subagents with clear responsibilities and appropriate tool access, you can build sophisticated AI workflows that maintain context clarity while leveraging specialized expertise.