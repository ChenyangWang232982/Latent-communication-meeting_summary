"""Review output separation and transparent textual change logs."""
import difflib
import re

def split_review_response(response):
    """Separate optional explanations without rejecting free-form model output."""
    marker = re.search(r'(?im)^\s*(?:#{1,6}\s*)?\*{0,2}Review reasons\s*:?\*{0,2}\s*$', response)
    if marker:
        return response[:marker.start()].strip(), response[marker.end():].strip()
    return response.strip(), ''


def change_log(draft, organized, reasons):
    lines = ['Summary organization change log',
             'Textual differences and model-provided reasons; evidence review is not independently verified.', '',
             *difflib.unified_diff(draft.splitlines(), organized.splitlines(),
                                   fromfile='draft', tofile='organized', lineterm=''),
             '', 'Model-provided reasons (not independently verified):']
    lines.append(reasons or 'No separate reasons section was identified in the model output.')
    return '\n'.join(lines) + '\n'
