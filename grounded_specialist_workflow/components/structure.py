"""Minimal JSON decoding for topic discovery and specialist reports."""
import json



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



def parse_topics(text):
    topics = parse_object(text)['topics']
    return [dict(id=f'topic_{i:03d}', title=t['title'], scope=t.get('scope', t['title']))
            for i, t in enumerate(topics)]
