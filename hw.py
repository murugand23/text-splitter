"""
Gamma AI Engineer Homework Assignment

Split a markdown document into N slides (1-50) while preserving all content.

Approach:
1. Split document into paragraphs (semantic chunks)
2. LLM analyzes paragraphs and groups them into slides based on discrete ideas
3. Iterative refinement ensures exact slide count
4. Return slides at paragraph boundaries (ensures content preservation)
"""

import os
from dotenv import load_dotenv
import json
import re
from typing import List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import tiktoken

load_dotenv()

# Global cost tracking
TOTAL_TOKENS_USED = 0
COST_PER_MILLION_TOKENS = 0.15  # gpt-4o-mini pricing


def count_tokens(text: str) -> int:
    """Count tokens exactly using tiktoken (cl100k_base encoding)."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def split_into_paragraphs(text: str) -> Tuple[List[str], List[int]]:
    """
    Split text into paragraphs and track their character positions.
    
    Splits by double newlines (\n\n) to get semantic chunks.
    Returns (paragraphs, positions) for content reconstruction.
    """
    paragraphs = []
    positions = []
    current_pos = 0
    parts = text.split('\n\n')
    
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            current_pos += len(part) + 2  # +2 for \n\n
            continue
        
        # Find this paragraph in the original text
        idx = text.find(part_stripped, current_pos)
        if idx == -1:
            raise RuntimeError(f"Content preservation error: paragraph not found in original text")
        
        paragraphs.append(part_stripped)
        positions.append(idx)
        current_pos = idx + len(part_stripped)
    
    return paragraphs, positions


def split_into_sentences(text: str) -> Tuple[List[str], List[int]]:
    """
    Split text into sentences using simple regex.
    Fallback for documents with too few paragraphs.
    Returns (sentences, positions) for content reconstruction.
    """
    # Simple sentence boundary: '. ', '! ', '? ' followed by capital letter or end
    pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    sentences = []
    positions = []
    current_pos = 0
    
    parts = pattern.split(text)
    
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue
        
        idx = text.find(part_stripped, current_pos)
        if idx == -1:
            raise RuntimeError("Content preservation error: sentence not found in original text")
        
        sentences.append(part_stripped)
        positions.append(idx)
        current_pos = idx + len(part_stripped)
    
    return sentences, positions


def truncate_smart(text: str, max_chars: int = 500) -> str:
    """Smart truncate: keep first 400 chars and last 100 chars."""
    if len(text) <= max_chars:
        return text
    return text[:400] + "..." + text[-100:]


def split_at_positions(text: str, positions: List[int]) -> List[str]:
    """Split text at given character positions."""
    if not positions:
        return [text]
    
    positions = sorted(set(p for p in positions if 0 < p < len(text)))
    split_points = [0] + positions + [len(text)]
    
    chunks = []
    for i in range(len(split_points) - 1):
        chunk = text[split_points[i]:split_points[i + 1]]
        if chunk:
            chunks.append(chunk)
    
    return chunks


def llm_refine_add_split(paragraphs: List[str], section_start: int, section_end: int, 
                         client, max_retries: int = 2) -> Optional[int]:
    """
    Ask LLM to split one section into 2 parts.
    Returns the paragraph number where the split should occur, or None if all retries fail.
    """
    global TOTAL_TOKENS_USED
    
    section_paras = paragraphs[section_start:section_end]
    section_tokens = sum(count_tokens(p) for p in section_paras)
    
    para_list = []
    for i, p in enumerate(section_paras):
        para_idx = section_start + i
        para_tokens = count_tokens(p)
        para_list.append(f"[{para_idx}] (~{para_tokens} tokens) {truncate_smart(p)}")
    
    prompt = f"""
This section has {len(section_paras)} paragraphs (~{section_tokens} tokens) and needs to be split into 2 parts.

Identify the best semantic boundary where the topic shifts.
If you see a markdown heading (starts with #), split BEFORE it to keep the heading with its content.

Return the paragraph number where the FIRST part should END.

Paragraphs (numbered {section_start} to {section_end - 1}):
{chr(10).join(para_list)}

Return JSON: {{"split_after_paragraph": X}}
""".strip()
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            
            TOTAL_TOKENS_USED += resp.usage.total_tokens
            data = json.loads(resp.choices[0].message.content)
            split_pos = data.get("split_after_paragraph")
            
            if isinstance(split_pos, int) and section_start <= split_pos < section_end - 1:
                return split_pos
            else:
                print(f"[hw.py] Attempt {attempt + 1}: LLM suggested invalid split {split_pos}")
                
        except Exception as e:
            print(f"[hw.py] Attempt {attempt + 1} failed: {e}")
    
    return None  # All retries failed


def llm_refine_remove_split(paragraphs: List[str], current_splits: List[int], 
                            client, max_retries: int = 2) -> Optional[int]:
    """
    Ask LLM which split to remove to maintain best semantic quality.
    Returns the paragraph number of the split to remove, or None if all retries fail.
    """
    global TOTAL_TOKENS_USED
    
    split_points = [-1] + sorted(current_splits) + [len(paragraphs) - 1]
    sections_info = []
    
    for i in range(len(split_points) - 1):
        start = split_points[i] + 1
        end = split_points[i + 1]
        section_size = end - start + 1
        section_tokens = sum(count_tokens(paragraphs[j]) for j in range(start, min(end + 1, len(paragraphs))))
        
        first_para = truncate_smart(paragraphs[start]) if start < len(paragraphs) else ""
        last_para = truncate_smart(paragraphs[end]) if end < len(paragraphs) else ""
        
        sections_info.append(
            f"Section {i+1}: paragraphs [{start}:{end}], {section_size} paras (~{section_tokens} tokens)\n"
            f"  First: {first_para}\n"
            f"  Last: {last_para}"
        )
    
    prompt = f"""
We have {len(current_splits) + 1} sections but need {len(current_splits)} (remove 1 split).

Current sections:
{chr(10).join(sections_info)}

Which split should be REMOVED to maintain the best semantic grouping?

Consider:
- Keep related content together (lists, examples, explanations)
- Avoid orphaning headings
- Prefer merging smaller adjacent sections
- Maintain logical flow

Current splits are at paragraphs: {current_splits}

Return the paragraph number of the split to REMOVE: {{"remove_split_at_paragraph": X}}
""".strip()
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            TOTAL_TOKENS_USED += resp.usage.total_tokens
            result = json.loads(resp.choices[0].message.content)
            split_to_remove = result.get("remove_split_at_paragraph")
            
            if split_to_remove in current_splits:
                return split_to_remove
            else:
                print(f"[hw.py] Attempt {attempt + 1}: LLM suggested invalid split {split_to_remove}")
                
        except Exception as e:
            print(f"[hw.py] Attempt {attempt + 1} failed: {e}")
    
    return None  # All retries failed


def iterative_refinement(paragraphs: List[str], initial_splits: List[int], 
                        target_count: int, client) -> List[int]:
    """
    Iteratively refine splits to reach exact count using LLM-driven decisions.
    Max 10 iterations to add/remove splits.
    """
    splits = list(initial_splits)
    
    for iteration in range(10):
        diff = target_count - len(splits)
        
        if diff == 0:
            print(f"[hw.py] ✓ Exact count after {iteration} refinements")
            return splits
        
        elif diff > 0:
            # Add split in largest section (by tokens) using LLM
            split_points = [-1] + sorted(splits) + [len(paragraphs) - 1]
            sections = []
            for i in range(len(split_points) - 1):
                start, end = split_points[i] + 1, split_points[i + 1] + 1
                if end - start > 1:
                    section_tokens = sum(count_tokens(paragraphs[j]) for j in range(start, end))
                    sections.append((section_tokens, start, end))
            
            if not sections:
                print(f"[hw.py] ⚠️  Cannot add more splits (no splittable sections)")
                return splits
            
            _, start, end = max(sections)
            print(f"[hw.py] Refinement {iteration + 1}: Adding split in [{start}:{end}]")
            new_split = llm_refine_add_split(paragraphs, start, end, client)
            
            if new_split is None:
                print(f"[hw.py] ⚠️  Cannot add split (LLM failed after retries)")
                return splits
            
            splits.append(new_split)
            splits.sort()
        
        else:
            # Remove split using LLM
            print(f"[hw.py] Refinement {iteration + 1}: Asking LLM which split to remove")
            split_to_remove = llm_refine_remove_split(paragraphs, splits, client)
            
            if split_to_remove is None:
                print(f"[hw.py] ⚠️  Cannot remove split (LLM failed after retries)")
                return splits
            
            splits.remove(split_to_remove)
            print(f"[hw.py] Removed split at paragraph {split_to_remove}")
    
    print(f"[hw.py] ⚠️  Max iterations reached")
    return splits


def llm_suggest_splits(text: str, target_slides: int) -> Tuple[Optional[List[int]], dict]:
    """
    Ask LLM to group paragraphs into slides based on discrete ideas.
    Returns (character positions, metadata) or (None, metadata) if fails.
    """
    metadata = {
        'llm_success': False,
        'attempts': 0,
        'llm_returned_count': 0
    }
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        print("[hw.py] No API key found")
        return None, metadata
    
    # Split into paragraphs
    paragraphs, para_positions = split_into_paragraphs(text)
    
    # Fallback: if not enough paragraphs, use sentence-level splitting
    if len(paragraphs) < target_slides:
        print(f"[hw.py] Only {len(paragraphs)} paragraphs for {target_slides} slides, using sentence-level splitting")
        paragraphs, para_positions = split_into_sentences(text)
        
        # Final check: if still not enough sentences
        if len(paragraphs) < target_slides:
            print(f"[hw.py] Only {len(paragraphs)} sentences for {target_slides} slides")
            return None, metadata
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Format paragraphs
        para_lines = [f"[{i}] {truncate_smart(p)}" for i, p in enumerate(paragraphs)]
        para_summary = "\n".join(para_lines)
        
        user_msg = f"""
Split this document into {target_slides} slides by grouping paragraphs into discrete ideas.

CRITICAL: Read through ALL {len(paragraphs)} paragraphs BEFORE deciding where to split.
- First pass: Identify all topic boundaries throughout the ENTIRE document
- Second pass: Select split points that create a coherent presentation

QUALITY GUIDELINES:
1. Each slide = ONE discrete idea
2. Split at topic/subtopic boundaries
3. If a paragraph starts with a markdown heading (#, ##, ###), split BEFORE it to keep the heading with its content
4. Avoid mid-thought splits
5. Create natural flow for a presentation
6. Distribute content reasonably

PROCESS:
1. Read ALL paragraphs below (0 to {len(paragraphs)-1})
2. Identify natural topic boundaries
3. Select exactly {target_slides - 1} split points
4. VERIFY: Don't split at a heading - split before it

DOCUMENT INFO:
- Total paragraphs: {len(paragraphs)}
- Target slides: {target_slides}

Paragraphs:
{para_summary}

Return JSON with paragraph numbers where slides should end:
{{"slide_ends_after_paragraphs": [...]}}
""".strip()
        
        global TOTAL_TOKENS_USED
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at splitting documents into semantic sections. Each section should represent ONE discrete idea. Respond with valid JSON only."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        
        TOTAL_TOKENS_USED += resp.usage.total_tokens
        data = json.loads(resp.choices[0].message.content)
        para_nums = data.get("slide_ends_after_paragraphs", [])
        
        # Validate and deduplicate
        validated = sorted(set(s for s in para_nums 
                              if isinstance(s, int) and 0 <= s < len(paragraphs) - 1))
        
        expected_count = target_slides - 1
        metadata['attempts'] = 1
        metadata['llm_returned_count'] = len(validated)
        
        print(f"[hw.py] LLM returned {len(validated)} splits, target is {expected_count}")
        
        # Use iterative refinement to adjust to exact count
        if len(validated) != expected_count:
            print(f"[hw.py] Using iterative refinement to adjust count")
            validated = iterative_refinement(paragraphs, validated, expected_count, client)
            metadata['attempts'] += 1
        
        if not validated or len(validated) != expected_count:
            print(f"[hw.py] ❌ Failed to achieve exact count")
            return None, metadata
        
        metadata['llm_success'] = True
        print(f"[hw.py] ✓ Achieved exact count: {len(validated)}")
        
        # Convert paragraph numbers to character positions
        split_positions = [para_positions[s + 1] for s in validated if s + 1 < len(para_positions)]
        
        return split_positions, metadata
        
    except Exception as e:
        print(f"[hw.py] LLM split failed: {e}")
        return None, metadata


def split_document(document: str, target_slides: int) -> Tuple[List[str], dict]:
    """
    Split a markdown document into target number of slides.
    
    Args:
        document: Markdown document as string
        target_slides: Target number of slides (1-50)
    
    Returns:
        Tuple of (slides, metadata) where metadata contains:
        - llm_success: bool
        - attempts: int
        - tokens_used: int
        - cost: float
    """
    metadata = {
        'llm_success': False,
        'attempts': 0,
        'tokens_used': 0,
        'cost': 0.0
    }
    
    # Validate inputs
    if not document:
        return [""], metadata
    
    if target_slides < 1 or target_slides > 50:
        raise ValueError("target_slides must be between 1 and 50")
    
    if target_slides == 1:
        return [document], metadata
    
    # Try LLM splitting
    llm_positions, llm_metadata = llm_suggest_splits(document, target_slides)
    metadata.update(llm_metadata)
    
    if not llm_positions:
        raise RuntimeError(
            "Failed to split document. Please ensure:\n"
            "1. OPENAI_API_KEY environment variable is set\n"
            "2. Document has sufficient content (at least 20 tokens per slide)\n"
            "3. Document has enough paragraphs for target slides"
        )
    
    # Split at LLM-suggested positions
    slides = split_at_positions(document, llm_positions)
    print(f"[hw.py] LLM suggested {len(slides)} chunks (target: {target_slides})")
    
    # Validate content preservation
    joined = ''.join(slides)
    if joined != document:
        print(f"⚠️  Warning: Content preservation check failed!")
        print(f"   Original: {len(document)} chars, Joined: {len(joined)} chars")
    
    # Cost tracking
    cost = TOTAL_TOKENS_USED * COST_PER_MILLION_TOKENS / 1_000_000
    metadata['tokens_used'] = TOTAL_TOKENS_USED
    metadata['cost'] = cost
    print(f"[hw.py] Total tokens used: {TOTAL_TOKENS_USED:,} | Estimated cost: ${cost:.4f}")
    
    return slides, metadata
