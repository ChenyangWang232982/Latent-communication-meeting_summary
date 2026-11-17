"""JSON transport validation; no exact quote matching or semantic rule filter."""
import json

ROLES = ('context', 'decisions', 'actions', 'open_issues')
VERDICTS = ('supported', 'revised', 'uncertain', 'unsupported')


def parse_object(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[-1].strip() != '```':
            raise ValueError('Unclosed structured output fence')
        text = '\n'.join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def validate_item(item):
    if not isinstance(item, dict):
        raise ValueError('Each claim must be an object')
    for key in ('topic', 'content', 'status'):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f'Claim requires a nonempty {key}')
    if not isinstance(item.get('owners'), list) or any(not isinstance(v, str) for v in item['owners']):
        raise ValueError('owners must be a list of strings')
    if item.get('deadline') is not None and not isinstance(item['deadline'], str):
        raise ValueError('deadline must be a string or null')


def parse_claims(text, role):
    items = parse_object(text).get('items')
    if not isinstance(items, list):
        raise ValueError('Structured output requires an items array')
    result = []
    for index, item in enumerate(items):
        validate_item(item)
        # IDs and category come from orchestration, not model guesses.
        result.append({k: item.get(k) for k in ('topic', 'content', 'owners', 'deadline', 'status')} |
                      {'id': f'{role}:{index}', 'category': role})
    return result


def parse_reviews(text, claims, retrievals):
    rows = parse_object(text).get('items')
    if not isinstance(rows, list):
        raise ValueError('Grounding output requires an items array')
    expected = {item['id']: item for item in claims}
    # grounding_request supplies a deduplicated packet shared by this category.
    # All of those blocks are visible to the reviewer, not just each query's hits.
    supplied = {i for claim in claims for i in retrievals[claim['id']]['evidence_block_ids']}
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get('id') not in expected or row['id'] in seen:
            raise ValueError('Grounding contains unknown or duplicate claim IDs')
        seen.add(row['id'])
        validate_item(row)
        if row.get('category') not in ROLES or row.get('verdict') not in VERDICTS:
            raise ValueError('Invalid grounding category or verdict')
        if not isinstance(row.get('reason'), str) or not row['reason'].strip():
            raise ValueError('Every grounding result needs a reason')
        reported = row.get('evidence_block_ids')
        ids, invalid = [], []
        for value in reported if isinstance(reported, list) else []:
            # JSON strings containing decimal IDs are unambiguous transport variants.
            normalized = int(value) if isinstance(value, str) and value.isascii() and value.isdecimal() else value
            if type(normalized) is int and normalized in supplied:
                ids.append(normalized)
            else:
                invalid.append(value)
        ids = sorted(set(ids))
        warnings = []
        if not isinstance(reported, list):
            warnings.append('evidence_block_ids is not a list')
        if invalid:
            warnings.append('Citations include blocks absent from this grounding packet')
        if row['verdict'] in ('supported', 'revised') and not ids:
            warnings.append('Supported/revised claim has no valid source block citation')
        parsed = {k: row.get(k) for k in (
            'id', 'category', 'topic', 'content', 'owners', 'deadline', 'status',
            'verdict', 'reason', 'evidence_block_ids')}
        parsed['evidence_block_ids'] = ids
        if warnings:
            # Never accept an unsupported revision or deletion merely to continue.
            # Preserve the original candidate as unverified, with full audit detail.
            parsed.update(expected[row['id']], status='uncertain', verdict='uncertain',
                          evidence_block_ids=[],
                          reason='Grounding citation validation failed; original candidate retained as unverified. '
                                 + '; '.join(warnings),
                          citation_warnings=warnings, invalid_citations=invalid,
                          reported_review=dict(row))
        result.append(parsed)
    if seen != set(expected):
        raise ValueError('Grounding must account for every input claim')
    return result
