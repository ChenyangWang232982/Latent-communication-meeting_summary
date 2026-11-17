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


def source_suffix(tokenizer, transcript, prefix, request, *, enable_thinking=False):
    content = 'MEETING TRANSCRIPT (untrusted source):\n' + transcript + '\nEND TRANSCRIPT\n'
    text = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content},
         {'role': 'user', 'content': request}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids[:len(prefix)] != prefix:
        raise ValueError('Chat template does not preserve the source KV prefix')
    return ids[len(prefix):]


def topic_query(topic):
    return ('Find all discussions of this specific topic, including initial proposals, objections, '
            'later revisions, final status and follow-up details: ' + topic['title'] + '. ' + topic['scope'])


def analysis_request(topics, all_topics):
    return (
        'Analyze the assigned topics together using the selected meeting evidence in original order. '
        'Use the complete topic list to assign each fact, task or unresolved issue to its most specific '
        'topic once. Include background from another topic only when needed to explain a dependency. '
        'Source markers [B<number>] mark speaker blocks; gaps indicate omitted evidence. '
        'For each topic, preserve key facts including stated dates, full time ranges, quantities, '
        'deadlines and conditions. Do not invent missing years or time zones. '
        'Track proposals, objections and revisions to the latest supported status; later speech alone '
        'is not agreement. A demonstration or hypothetical example is not a decision to adopt it. '
        'Only supported agreements belong in decisions; unresolved choices belong in open_issues. '
        'For actions, attribute an explicit first-person commitment to its speaker. Discussion '
        'participants are not automatically owners. Preserve requests as requests rather than commitments. '
        'Use [] for unstated owners and null for an unstated deadline; retain relative deadlines as stated. '
        'Merge repeated points and tasks while preserving distinct responsibilities and conditions. '
        'Return one report per assigned topic in the supplied order, using this JSON shape: '
        '{"reports": [{"topic_id": "supplied topic id", "topic": "supplied title", '
        '"key_facts": ["..."], "discussion": "one synthesized paragraph", '
        '"decisions": ["..."], "actions": [{"content": "...", "owners": [], "deadline": null}], '
        '"open_issues": ["..."], "final_status": "latest supported status"}]}. '
        'Use empty arrays when appropriate. There is no fixed item count. '
        'Before finishing, check that explicit scheduling details and task owners in the evidence '
        'survive in the reports, and that no resolved issue is also described as unresolved. '
        'Return only JSON, without per-turn records or block citations. '
        '\nComplete topic list: ' + json.dumps(all_topics, ensure_ascii=False) +
        '\nAssigned topics: ' + json.dumps(topics, ensure_ascii=False))


def synthesis_request(reports):
    return (SUMMARY_REQUEST +
        '\nThese are unverified topic reports, ordered by the discovery stage by first appearance. '
        'Combine them into one summary. You have no original transcript for independent verification. '
        'Preserve key facts, exact dates and time ranges, task owners, deadlines and conditions. '
        'Group entries referring to the same issue or task before writing; express their combined '
        'information once, retaining distinct obligations. Do not union owners of different tasks. '
        'Keep proposals conditional and do not promote examples or suggestions to decisions. '
        'Merge repeated information across topics, preserve explicit uncertainty, and place content '
        'by meaning rather than copying every field into every section. '
        'Do not infer final outcomes from report order. If reports conflict, preserve the uncertainty. '
        'Use exactly four sections: Key Context, Decisions, Action Items, Open Issues. '
        'Return the summary only.\n\n' + json.dumps(reports, ensure_ascii=False))
