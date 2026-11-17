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
        ids = row.get('evidence_block_ids')
        allowed = retrievals[row['id']]['evidence_block_ids']
        if not isinstance(ids, list) or any(type(i) is not int or i not in allowed for i in ids):
            raise ValueError('Grounding cites a block not supplied for this claim')
        if row['verdict'] in ('supported', 'revised') and not ids:
            raise ValueError('Supported/revised claims need source block references')
        result.append({k: row.get(k) for k in (
            'id', 'category', 'topic', 'content', 'owners', 'deadline', 'status',
            'verdict', 'reason', 'evidence_block_ids')})
    if seen != set(expected):
        raise ValueError('Grounding must account for every input claim')
    return result
