"""Minimal JSON decoding for topic discovery and specialist reports."""
import json


def split_thinking(tokenizer, token_ids, terminal_tokens):
    """Separate Qwen's generated reasoning from JSON using actual generated IDs."""
    content = token_ids[:-terminal_tokens] if terminal_tokens else token_ids
    closing = tokenizer.encode('</think>', add_special_tokens=False)
    for index in range(len(content) - len(closing) + 1):
        if closing and content[index:index+len(closing)] == closing:
            answer = content[index+len(closing):]
            return (tokenizer.decode(answer, skip_special_tokens=True).strip(),
                    tokenizer.decode(content[:index], skip_special_tokens=True).strip(),
                    dict(thinking_tokens=index, delimiter_tokens=len(closing),
                         answer_tokens=len(answer), terminal_tokens=terminal_tokens, thinking_complete=True))
    raise ValueError('Specialist thinking did not emit </think>; raw generated output saved')



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
