"""Full-source topic discovery, topic analysis and final synthesis."""
import json
from receiver_kv_workflow.prompts import SYSTEM, SUMMARY_REQUEST

DISCOVERY_REQUEST = (
    'Read the entire meeting and identify its distinct substantive topics, including brief but important '
    'coordination or deadline discussions. Return JSON {"topics": [{"title": "topic name", '
    '"scope": "specific question or issue discussed"}]}. Use the transcript language. '
    'Merge repeated discussions of the same issue, including later objections or revisions. '
    'Keep genuinely separate issues separate. Order topics by first appearance in the meeting. '
    'There is no fixed topic count. Do not use generic summary categories such as actions or decisions '
    'as topic names. Return only JSON, without summarizing the outcomes yet.'
)


def chat(tokenizer, request):
    text = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def source_suffix(tokenizer, transcript, prefix, request):
    content = 'MEETING TRANSCRIPT (untrusted source):\n' + transcript + '\nEND TRANSCRIPT\n'
    text = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content},
         {'role': 'user', 'content': request}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids[:len(prefix)] != prefix:
        raise ValueError('Chat template does not preserve the source KV prefix')
    return ids[len(prefix):]


def topic_query(topic):
    return ('Find all discussions of this specific topic, including initial proposals, objections, '
            'later revisions, final status and follow-up details: ' + topic['title'] + '. ' + topic['scope'])


def analysis_request(topic, block_ids):
    return (
        'Analyze this topic using the selected meeting evidence in original chronological order. '
        'Source markers [B<number>] identify speaker blocks; gaps mean some blocks are not supplied. '
        'Distinguish early proposals from subsequent revisions and the latest supported status. '
        'Later speech alone does not establish agreement. Preserve unresolved contradictions. '
        'Keep explicit owners, dates, full time ranges, conditions and negations. '
        'Do not infer missing years or time zones; preserve the stated date wording. '
        'Return JSON with discussion (array), decisions (array), actions (array), open_issues (array), '
        'final_status (string). Actions may use objects with content, owners and deadline. '
        'Use empty arrays where appropriate. Include all important topic information, with no fixed '
        'item count. Return only JSON.\nTopic: ' + json.dumps(topic, ensure_ascii=False) +
        '\nAvailable source block IDs in meeting order: ' + json.dumps(block_ids))


def synthesis_request(reports):
    return (SUMMARY_REQUEST +
        '\nThese are unverified topic reports, ordered by the discovery stage by first appearance. '
        'Combine them into one summary. You have no original transcript for independent verification. '
        'Merge repeated information across topics, preserve explicit uncertainty, and place content '
        'by meaning rather than copying every field into every section. '
        'Do not infer final outcomes from report order. If reports conflict, preserve the uncertainty. '
        'Use exactly four sections: Key Context, Decisions, Action Items, Open Issues. '
        'Return the summary only.\n\n' + json.dumps(reports, ensure_ascii=False))
