# Claude Code AI Development Agents

This directory contains specialized AI development agents for the gogooku3-standalone project.

## Available Agents

### 1. ai-model-developer
**Expert in PyTorch model development and architecture design**

Use this agent when:
- Developing new ML model architectures
- Modifying ATFT-GAT-FAN model components
- Implementing custom loss functions
- Debugging model issues (NaN gradients, convergence problems)
- Optimizing model performance
- Adding new model features or layers

Example invocations:
- "Help me improve the ATFT-GAT-FAN model performance"
- "Debug why the model has NaN gradients"
- "Add a new attention mechanism to the model"
- "Optimize GPU utilization for the model"

### 2. data-pipeline-engineer
**Expert in data pipeline, ETL, and feature engineering**

Use this agent when:
- Building or optimizing data pipelines
- Working with JQuants API integration
- Adding new financial features
- Handling data quality issues
- Optimizing pipeline performance
- Ensuring temporal data safety

Example invocations:
- "Add momentum features to the pipeline"
- "Optimize the data pipeline for faster processing"
- "Fix missing data issues in the dataset"
- "Enable GPU-accelerated ETL"

### 3. training-optimizer
**Expert in model training, optimization, and experimentation**

Use this agent when:
- Setting up or debugging training pipelines
- Hyperparameter tuning
- Fixing training issues (overfitting, slow convergence)
- Optimizing training performance
- Managing experiments and checkpoints
- Implementing learning rate schedules

Example invocations:
- "Training is too slow, optimize it"
- "Model is overfitting, add regularization"
- "Set up hyperparameter optimization"
- "Fix unstable training with exploding gradients"

## How to Use

### Method 1: Implicit Invocation (Recommended)
Claude Code will automatically select the appropriate agent based on your request context.

```
User: "Improve model convergence speed"
→ Claude Code automatically uses 'training-optimizer' agent
```

### Method 2: Explicit Invocation
Mention the agent type in your request:

```
User: "@ai-model-developer help me add a new loss function"
```

### Method 3: Using /agents Command
Use the `/agents` command to:
- List available agents: `/agents list`
- View agent details: `/agents show ai-model-developer`
- Manage agents: `/agents`

## Agent Specialization

Each agent has:
- **Deep project knowledge**: Specific file paths, configurations, and conventions
- **Specialized tools**: Access to Read, Write, Edit, Glob, Grep, Bash
- **Best practices**: Safety rules, optimization techniques, debugging strategies
- **Example workflows**: Common tasks and solutions

## Integration with Project

All agents are configured with:
- **gogooku3-standalone** project structure
- **ATFT-GAT-FAN** architecture details
- **Training infrastructure** (Hydra configs, make commands)
- **Data pipeline** (up to 395 features, ~307 active; JQuants API, GPU-ETL)
- **Safety mechanisms** (Walk-Forward validation, cross-sectional normalization)

## Best Practices

1. **Use the right agent**: Choose based on your task domain
2. **Be specific**: Provide context about what you're trying to achieve
3. **Iterative approach**: Start with diagnosis, then implement, then validate
4. **Safety first**: Agents prioritize data safety and temporal integrity
5. **Performance aware**: Agents consider GPU/memory constraints

## Examples

### Model Development
```
User: "Add residual connections to the FAN module"
→ Uses ai-model-developer
→ Analyzes src/gogooku3/models/architectures/atft_gat_fan.py
→ Implements changes with proper forward pass
→ Updates config if needed
→ Suggests testing with make smoke
```

### Data Pipeline
```
User: "Pipeline is taking 60 minutes, make it faster"
→ Uses data-pipeline-engineer
→ Profiles current pipeline
→ Enables GPU-ETL, optimizes Polars usage
→ Reduces time to 15-20 minutes
```

### Training
```
User: "Model overfits after 30 epochs"
→ Uses training-optimizer
→ Analyzes train/val gap
→ Adds dropout, early stopping, weight decay
→ Monitors convergence with new settings
```

## Notes

- Agents have **separate context windows** from the main conversation
- Agents can **chain together** for complex tasks
- All agents respect **project safety constraints** (no future leakage, temporal ordering)
- Agents are **version controlled** - customize for your team's workflow

## Support

For questions or issues:
- Check agent system prompts in the .md files
- Use `/agents` command to inspect configurations
- Refer to Claude Code documentation: https://docs.claude.com/en/docs/claude-code/sub-agents.md
