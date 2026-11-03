"""Basic structural checks and a transparent edit log for review outputs."""
import difflib
import json
import re

SECTIONS = ('key context', 'decisions', 'action items', 'open issues')


def section_names(text):
    found = set()
    for line in text.splitlines():
        clean = re.sub(r'^[\s#*]+|[\s*:#]+$', '', line).strip().casefold()
        if clean in SECTIONS:
            found.add(clean)
    return found


def missing_sections(draft, candidate):
    return sorted(section_names(draft) - section_names(candidate))


def apply_evidence_edits(draft, response):
    """Apply only explicitly justified one-line replacements matched exactly once."""
    response = response.strip()
    if response.startswith('```'):
        response = re.sub(r'^```(?:json)?\s*|\s*```$', '', response, flags=re.I).strip()
    data = json.loads(response)
    if not isinstance(data, dict) or not isinstance(data.get('edits'), list):
        raise ValueError('Expected an object containing an edits array')
    lines = draft.splitlines()
    original_lines = list(lines)
    edits = data['edits']
    used = set()
    for item in edits:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip()
                                              for k in ('before', 'after', 'reason', 'evidence_basis')):
            raise ValueError('Every edit requires before, after, reason and evidence_basis')
        before, after = item['before'], item['after']
        if '\n' in before or '\n' in after or not re.match(r'^\s*(?:[-*•]\s+|\d+[.)]\s+)', before):
            raise ValueError('Edits must replace a single complete bullet line')
        if not re.match(r'^\s*(?:[-*•]\s+|\d+[.)]\s+)', after) or before == after:
            raise ValueError('Replacement must be a changed, nonempty bullet')
        if before in used or original_lines.count(before) != 1:
            raise ValueError('Original bullet is missing, ambiguous, or repeated')
        used.add(before)
        lines[original_lines.index(before)] = after
    return '\n'.join(lines), edits


def change_log(draft, organized, reviewed, retrieval, edits):
    lines = ['Review edit log',
             'Changes are textual diffs, not verified facts.',
             'Retrieved passages are candidate evidence, not verified claim-level citations.', '']
    for title, before, after in (('Organization', draft, organized), ('Evidence review', organized, reviewed)):
        lines += [title, *difflib.unified_diff(before.splitlines(), after.splitlines(),
                  fromfile='before', tofile='after', lineterm=''), '']
    lines.append(f"Retrieval: {retrieval['matched_block_count']} matched blocks; "
                 f"{retrieval['source_kv_positions']} source KV positions.")
    lines += ['', 'Model-provided reasons and evidence basis (not independently verified):']
    if not edits:
        lines.append('No evidence-review changes.')
    for index, edit in enumerate(edits, 1):
        lines += [f"{index}. Before: {edit['before']}", f"   After: {edit['after']}",
                  f"   Reason: {edit['reason']}", f"   Evidence basis: {edit['evidence_basis']}"]
    return '\n'.join(lines) + '\n'
