# Chart Interpretation Evaluation Methodology

## Overview

This document describes the comprehensive methodology used to evaluate vision model performance on chart interpretation tasks. The evaluation framework assesses three leading models (Claude Sonnet 4.5, GPT-5, and Gemini 2.5 Pro) across diverse business chart types with automated scoring and error detection.

## Evaluation Framework

### Chart Generation

**Chart Types and Distribution:**
- **General Business (4 charts)**: Revenue trends, KPI performance, conversion funnels, regional comparisons
- **Tech/Product (3 charts)**: User growth, A/B test results, latency monitoring
- **Financial (2 charts)**: Stock performance, budget allocation
- **Edge Cases (1 chart)**: Multi-axis combo charts

**Technical Specifications:**
- Image size: 1092x1092 pixels
- Format: PNG with 110 DPI
- Reproducible data using fixed random seeds
- Realistic business scenarios with proper labels and formatting

### Question Taxonomy

**Tier 1 - Factual Questions:**
- Direct data extraction (specific values, labels, categories)
- Objective answers with minimal interpretation required
- Examples: "What is the revenue in Q3?", "Which quarter had the highest sales?"

**Tier 2 - Pattern Questions:**
- Trend analysis and pattern recognition
- Comparative analysis and growth calculations
- Examples: "Describe the overall revenue trend", "What is the growth rate from Q1 to Q4?"

### Scoring Methodology

**Numeric Answer Scoring:**
- Extract all numbers from model response using regex patterns
- Check if any extracted number falls within tolerance range
- Tolerance: ±1% for precise values, ±5% for approximations
- Score: 1.0 if within tolerance, 0.0 otherwise

**Categorical Answer Scoring:**
- Case-insensitive substring matching
- Partial word matching for multi-word answers
- Score: 1.0 if expected answer found in response, 0.0 otherwise

**Trend Detection Scoring:**
- Keyword-based scoring using predefined term lists
- Increasing terms: "increase", "growth", "upward", "rising"
- Decreasing terms: "decrease", "decline", "downward", "falling"
- Score: Proportion of relevant keywords found (0-1)

## Error Taxonomy

### 1. Number Hallucinations
**Detection Method:**
- Extract all numeric values from model response
- Compare against ground truth data points and key facts
- Flag numbers not found within ±10% tolerance of any valid value

**Scoring:**
- Score = (valid numbers found) / (total numbers in response)
- Flags responses with hallucinated numbers

### 2. Axis Errors
**Detection Method:**
- Check for mentions of axis-related terms
- Assess specificity of axis information provided
- Flag responses with missing or incorrect axis labels/units

**Scoring:**
- Score = 1.0 if axis information is specific and detailed
- Score = 0.5 if axis mentioned but lacks specificity
- Score = 0.0 if no axis awareness detected

### 3. Trend Reversals
**Detection Method:**
- Identify contradictory trend keywords in same response
- Flag responses containing both increasing and decreasing terms
- Detect logical inconsistencies in trend descriptions

**Scoring:**
- Score = 0.0 if contradictory terms found
- Score = 1.0 if consistent trend language used

### 4. Overgeneralization
**Detection Method:**
- Count generic phrases: "generally", "appears to", "seems to", "might be"
- Count specific indicators: "exactly", "precisely", "specifically"
- Calculate ratio of generic to specific language

**Scoring:**
- Score = max(0, 1 - (generic_ratio × 2))
- Flags responses with excessive hedging language

## Model Configuration

### Claude Sonnet 4.5 (AWS Bedrock)
- **Provider**: AWS Bedrock Runtime
- **Model ID**: `anthropic.claude-sonnet-4-5-20250929-v1:0`
- **Pricing**: $3.75/$18.75 per MTok (input/output)
- **Max Tokens**: 4096
- **Timeout**: 3600 seconds

### GPT-5 (OpenAI)
- **Provider**: OpenAI API
- **Model ID**: `gpt-5`
- **Pricing**: $1.25/$10.0 per MTok (input/output)
- **Max Tokens**: 4096
- **Temperature**: 0.1

### Gemini 2.5 Pro (Google Cloud)
- **Provider**: Google Cloud Vertex AI
- **Model ID**: `gemini-2.5-pro`
- **Pricing**: $1.25/$5.0 per MTok (input/output)
- **Max Tokens**: 4096
- **Temperature**: 0.1

## Budget and Cost Management

**Budget Limits:**
- Total budget: $50.00 USD (hard limit)
- Per-model budget: $20.00 USD
- Warning threshold: 80% of budget

**Cost Tracking:**
- Real-time cost monitoring per request
- Automatic budget enforcement with hard stops
- Detailed cost breakdown by model and question type
- Intermediate result caching for resumption

## Evaluation Pipeline

### 1. Pre-evaluation Setup
- Load ground truth data from JSON
- Initialize all model instances
- Verify API connectivity and credentials
- Initialize cost tracker with budget limits

### 2. Evaluation Loop
For each chart → each model → each question:
1. Generate standardized prompt
2. Call model API with retry logic (3 attempts, exponential backoff)
3. Track cost and update budget tracker
4. Score response using appropriate method
5. Run all 4 error detection functions
6. Store results with metadata

### 3. Post-evaluation Analysis
- Convert results to structured DataFrame
- Calculate aggregate metrics by model and tier
- Generate error analysis and insights
- Save results locally and upload to S3

## Quality Assurance

**Reproducibility:**
- Fixed random seeds for chart generation
- Deterministic question generation
- Consistent prompt templates
- Version-controlled evaluation scripts

**Validation:**
- Manual spot-checking of generated charts
- Ground truth verification for sample questions
- Cross-validation of scoring functions
- Error detection accuracy testing

**Monitoring:**
- Comprehensive logging with timestamps
- Real-time progress tracking with tqdm
- Intermediate result saving every 10 evaluations
- Detailed error reporting and debugging

## Limitations and Considerations

**Chart Complexity:**
- Limited to 2D business charts
- No 3D or interactive visualizations
- Focus on common chart types

**Language Bias:**
- English-only evaluation
- Business terminology focus
- Western business context

**Scoring Limitations:**
- Automated scoring may miss nuanced interpretations
- Tolerance ranges may be too strict/lenient
- Error detection is heuristic-based

**Model Availability:**
- Dependent on API availability and rate limits
- Model versions may change over time
- Pricing may fluctuate

## Future Enhancements

**Expanded Chart Types:**
- Geographic visualizations
- Network diagrams
- Scientific plots
- Interactive dashboards

**Advanced Scoring:**
- Semantic similarity scoring
- Multi-step reasoning evaluation
- Context-aware error detection
- Human evaluation integration

**Scalability Improvements:**
- Parallel evaluation processing
- Distributed computing support
- Real-time evaluation streaming
- Automated report generation
