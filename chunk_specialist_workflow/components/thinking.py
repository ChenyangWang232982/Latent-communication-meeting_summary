"""Split Qwen reasoning using generated token IDs, without re-tokenizing output."""


def split_thinking(tokenizer, generated, terminal_tokens):
    content = generated[:-terminal_tokens] if terminal_tokens else generated
    close = tokenizer.encode('</think>', add_special_tokens=False)
    for index in range(len(content)-len(close)+1):
        if close and content[index:index+len(close)] == close:
            answer = content[index+len(close):]
            return (tokenizer.decode(answer, skip_special_tokens=True).strip(),
                    tokenizer.decode(content[:index], skip_special_tokens=True).strip(),
                    dict(thinking_tokens=index, delimiter_tokens=len(close), answer_tokens=len(answer),
                         terminal_tokens=terminal_tokens, thinking_complete=True))
    raise ValueError('Final summary did not close thinking with </think>; inspect live summary output')
