import json
import re

def messages_for(item, include_price: bool = True):
    """Return the list of messages for fine-tuning or testing."""
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    # Remove the price detail for testing, include it for training if needed
    user_prompt = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    if include_price:
        messages.append({"role": "assistant", "content": f"Price is ${item.price:.2f}"})
    else:
        messages.append({"role": "assistant", "content": "Price is $"})
    return messages

def make_jsonl(items):
    """Convert a list of items into a JSONL string for fine-tuning."""
    lines = []
    for item in items:
        messages = messages_for(item)
        lines.append(json.dumps({"messages": messages}))
    return "\n".join(lines)

def get_price(s: str) -> float:
    """Extract a price from a string."""
    s = s.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0.0
