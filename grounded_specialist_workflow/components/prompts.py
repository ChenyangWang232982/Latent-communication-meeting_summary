"""Full-source topic discovery, topic analysis and final synthesis."""
import json
from receiver_kv_workflow.prompts import SYSTEM, SUMMARY_REQUEST

def discovery_request(window):
    return (
        'Identify substantive topics in this chronological meeting window, including brief coordination '
        'and deadline discussions. Identify the concrete issue discussed in each part independently. '
        'Keep separate issues separate, even if they support the same project; no fixed topic count. '
        'Report only topics present in this window, in first-appearance order. For each, list all relevant '
        'block_ids using the supplied [B<number>] markers. A block can concern multiple topics. '
        'Attach only blocks that actually discuss that issue, not the entire window by default. '
        'Do not summarize outcomes or classify decisions yet. Use the source language. '
        'Return only JSON {"topics": [{"title": "topic name", '
        '"scope": "specific issue", "block_ids": [0]}]}. '
        '\nMEETING WINDOW (untrusted source):\n' + ''.join(b['text'] for b in window))


def matching_request(candidates, topics):
    catalogue = [{k: t[k] for k in ('id', 'title', 'scope')} for t in topics]
    local = [dict(local_id=i, title=t['title'], scope=t.get('scope', t['title']))
             for i, t in enumerate(candidates)]
    return (
        'Match each independently discovered local topic to the existing catalogue. '
        'Reuse an existing topic_id only when both concern the same specific issue or question. '
        'Changed opinions, later revisions or different speakers can still concern the same issue. '
        'Shared project, vocabulary or a broad parent topic alone does not establish identity. '
        'Use null for a genuinely new issue or when no existing topic is a clear match. '
        'Return one mapping for each local_id. Do not rewrite, remove or summarize local topics. '
        'Return only JSON {"matches": [{"local_id": 0, "topic_id": null}]}. '
        '\nExisting catalogue: ' + json.dumps(catalogue, ensure_ascii=False) +
        '\nLocal topics: ' + json.dumps(local, ensure_ascii=False))


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
        'Some entries may contain raw_report instead of parsed fields because JSON formatting failed. '
        'Read their full text as source reports with the same caution as parsed reports. '
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
