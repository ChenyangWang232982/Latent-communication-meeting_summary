"""Basic structural checks and a transparent edit log for review outputs."""
import difflib
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


def change_log(draft, organized, reviewed, retrieval):
    lines = ['Review edit log',
             'Changes are textual diffs, not verified facts.',
             'Retrieved passages are candidate evidence, not verified claim-level citations.', '']
    for title, before, after in (('Organization', draft, organized), ('Evidence review', organized, reviewed)):
        lines += [title, *difflib.unified_diff(before.splitlines(), after.splitlines(),
                  fromfile='before', tofile='after', lineterm=''), '']
    lines.append(f"Retrieval: {retrieval['matched_block_count']} matched blocks; "
                 f"{retrieval['source_kv_positions']} source KV positions.")
    return '\n'.join(lines) + '\n'
