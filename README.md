# Document Splitting with LLM

Splits markdown documents into N slides where each slide represents a discrete idea.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run evaluation on test files
python eval_framework.py
```

**Test your own files:** Edit `test_files` in `eval_framework.py` (line 152):

```python
test_files = [('your-file.md', 10)]  # (filename, target_slides)
```

## Approach

```
1. Split document into paragraphs (by \n\n), fallback to sentences if needed
2. Ask LLM (gpt-4o-mini) to group chunks into slides based on semantic boundaries
3. If count is wrong, use iterative refinement (LLM adds/removes splits)
4. Split document at the suggested character positions
```

**Why LLM?** The assignment requires "discrete ideas" - a semantic concept that needs content understanding, not just mechanical splitting.

## Architecture

### Key Functions

**`count_tokens(text) -> int`**

- Uses `tiktoken` with `cl100k_base` encoding for exact token counting
- Required for accurate cost tracking and length calculations

**`split_into_paragraphs(text) -> (paragraphs, positions)`**

- Splits by `\n\n`, tracks character positions for exact reconstruction
- Returns paragraph text + where each starts in original document
- Raises `RuntimeError` if content preservation fails (fail-fast)

**`split_into_sentences(text) -> (sentences, positions)`**

- Fallback for single-paragraph documents
- Uses regex-based sentence boundary detection
- Enables splitting when paragraphs < target_slides

**`truncate_smart(text, max_chars=500) -> str`**

- Truncates long paragraphs for LLM context (first 400 + last 100 chars)
- Balances cost vs context quality

**`llm_suggest_splits(text, target_slides) -> (positions, metadata)`**

- Main LLM orchestration using `gpt-4o-mini`
- Formats paragraphs as `[0] first 400 chars...last 100 chars` (truncated to 500 chars)
- Prompt includes semantic guidelines
- LLM returns paragraph numbers to split after (e.g., `[1, 3]` = split after paragraphs 1 and 3)
- Converts paragraph numbers to character positions in original text

**`llm_refine_add_split(paragraphs, section_start, section_end, client) -> Optional[int]`**

- Finds best place to add a split within a section
- Includes token counts in prompt for better decisions
- 2 retries at `temperature=0.3`, returns `None` on failure

**`llm_refine_remove_split(paragraphs, current_splits, client) -> Optional[int]`**

- Identifies which split to remove for best semantic quality
- Includes token counts in prompt for better decisions
- 2 retries at `temperature=0.3`, returns `None` on failure

**`iterative_refinement(paragraphs, splits, target, client) -> splits`**

- Adjusts to exact count (max 10 iterations)
- **Add split:** Picks largest section by tokens, asks LLM where to split
- **Remove split:** Asks LLM which split to remove for best semantic quality
- **No mechanical fallbacks:** Exits gracefully if LLM fails after retries (improved first-attempt rate 87% → 100%)

**`split_at_positions(text, positions) -> List[str]`**

- Splits text at exact character positions
- Ensures content preservation by using original text indices

**`split_document(document, target_slides) -> (slides, metadata)`**

- Top-level API
- Validates inputs, calls LLM, splits at positions
- Returns slides + metadata (cost, tokens, attempts)
- Tracks all LLM calls for cost/token accounting

### Character Position Tracking

The LLM returns paragraph indices (e.g., `[1, 3]` means "split after paragraph 1 and 3"). These are converted to character positions (where paragraphs 2 and 4 start in the original text). Splitting at these exact positions preserves all content, including whitespace.

## Design Decisions

- **Paragraph-first:** More semantic than sentences, keeps lists together, fewer LLM tokens
- **Sentence fallback:** Handles edge cases (single-paragraph docs)
- **Token-aware refinement:** Picks sections to split by token count, not paragraph count
- **500 char context:** Shows LLM enough text (400 start + 100 end) for good decisions
- **Retry logic:** 2 attempts per LLM call, no mechanical fallbacks
- **gpt-4o-mini:** 16x cheaper than gpt-4o, sufficient quality for semantic splitting
- **temperature=0.3:** Consistent but not robotic
- **Prompt engineering:** Heading placement, semantic guidelines in prompt (not post-processing)

## Performance

Based on latest eval run (3 documents, varying sizes):

**Correctness: 100%**

- Content preservation: ✅
- Exact count: ✅
- Heading accuracy: ~87%

**Cost: ~$0.0009/doc**

- Small (1-2k tokens): $0.0002-$0.0004
- Medium (3-5k tokens): $0.0009-$0.0013
- Large (10k+ tokens): $0.0018-$0.0034

**Speed: ~1.76s average**

**Quality:**

- Distribution: 1.97x max/avg (semantic grouping, not forced uniformity)
- First-attempt success: 100% (recent tests)

## Limitations

- **Uneven distribution (2x):** By design - prioritizes discrete ideas over uniform sizes
- **Heading accuracy (87%):** ~1 in 8 headings orphaned, trade-off for simplicity
- **Edge cases:** May fail if document too short for target slides

## Evaluation

The eval framework measures correctness, quality, cost, and speed across multiple test files.

```bash
python eval_framework.py
```

See metrics explanation below for what each measurement means.

### Metrics Measured

**Correctness (Must be 100%)**

- **Content Preservation:** `''.join(slides) == doc` - catches any data loss or whitespace issues
- **Exact Count:** `len(slides) == target_slides` - must hit exact count

**Quality**

- **Markdown Heading Accuracy:** % of markdown headings that stay with their content (not orphaned at end of slide)
  - Calculated: `(total_headings - orphaned) / total_headings × 100`
  - Current: ~87% (acceptable trade-off for simplicity)
- **Distribution:** How evenly content is distributed across slides
  - Calculated: `max(token_counts) / avg(token_counts)`
  - 1.0x = perfect uniformity, 2.0x = largest slide is 2x average
  - Current: ~1.97x (prioritizes semantic grouping over uniformity)

**Efficiency**

- **Cost:** Total $ spent on OpenAI API calls (tracks prompt + completion tokens)
  - Current: ~$0.0009/doc
- **Speed:** Wall-clock time from start to finish (includes all LLM calls)
  - Current: ~1.76s/doc
- **First-Attempt Success:** Did LLM return correct count on first try (no refinement needed)?
  - Current: 100% (better prompts = fewer retries = lower cost + faster speed)

Outputs detailed results including slide content, metrics, and cost analysis.

## Files

- `hw.py` (488 lines) - Main implementation
- `eval_framework.py` (163 lines) - Evaluation framework
- `README.md` - This file
