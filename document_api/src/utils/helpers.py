def format_response(decision, amount=None, justification=None):
    response = {
        "decision": decision,
        "amount": amount,
        "justification": justification
    }
    return response

def handle_error(message):
    return {
        "error": message
    }