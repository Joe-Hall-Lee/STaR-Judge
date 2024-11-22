import re


def _parse_output_absolute(output):
    # Check for "out of 5" or "/5"
    suffix_index = output.lower().find("out of 5")
    if suffix_index == -1:
        suffix_index = output.find("/5")
    if suffix_index == -1:
        suffix_index = output.find("]")

    if suffix_index != -1:
        score_part = output[:suffix_index].strip()
        match = re.search(r'(\d+)\s*$', score_part)
        if match:
            result = int(match.group(1))
            if 1 <= result <= 5:
                return output, result

    # If "out of 5" not found, look for "score of" or "scores a" or "score: " or "[" specifically
    prefix_index = output.lower().find("score of")
    prefix = "score of"
    if prefix_index == -1:
        prefix_index = output.lower().find("scores a")
        prefix = "scores a"
    if prefix_index == -1:
        prefix_index = output.lower().find("score:")
        prefix = "score:"

    if prefix_index != -1:
        score_part = output[prefix_index + len(prefix):].strip()
        match = re.search(r'(\d+)', score_part)
        if match:
            result = int(match.group(1))
            if 1 <= result <= 5:
                return output, result

    # Fallback to the more general regex
    pattern = r"""
    (?:                        # Start of a non-capturing group
        \[RESULT\]|\[SCORE\]|   # Match either "[RESULT]" or "[SCORE]"
        Score:?|score:?|        # Match "Score:" or "score:"
        Result:?|\[Result\]:?|  # Match "Result:" or "[Result]:"
        score\s+of|             # Match "score of"
        \[|\(                   # Match an opening bracket "[" or parenthesis "("
        |scores?\s?a?|           # Match "scores?" or "score a"
        |is\s+                   # Match "is "
    )                           # End of the non-capturing group
    \s*                         # Match any whitespace characters (including none)
    (\d+)                       # Capture one or more digits (the score)
    (?:                         # Start of another non-capturing group
        (?:\)|\]|\s*$)|         # Match either a closing bracket "]", parenthesis ")", 
                                # whitespace, or the end of the string "$"
        (?:/\s*5|               # Match "/ 5" with optional whitespace around the slash
           \s*out\s*of\s*5)     # or " out of 5 " with flexible whitespace around "out of"
        |of\s+5                  # or " of 5 " with a single space before "5"
    )?                          # End of the non-capturing group (optional)
    (?:\s*\.*\s*)               # Optionally match a period followed by any whitespace
    (?:.*)?                     # Optionally match any additional characters after the score
"""
    match = re.search(pattern, output, re.IGNORECASE | re.VERBOSE)

    if match:
        result = int(match.group(1))
        if 1 <= result <= 5:
            return output, result

    return None, None


def parse_output(outputs, mode: str):
    assert mode in [
        "absolute",
        "a2a"
    ]

    if mode == "absolute" or mode == "a2a":
        return _parse_output_absolute(outputs)


if __name__ == "__main__":
    # Test cases
    test_cases = [("Good job. [RESULT] 3", "a2a", 3),
                  ("Needs improvement. [RESULT] Score: 2", "a2a", 2),
                  ("Well done. [RESULT] Result: 4", "a2a", 4),
                  ("Average. [RESULT] 4/5", "a2a", 4),
                  ("Excellent. [RESULT] 5 out of 5", "a2a", 5),
                  ("Poor performance. [RESULT] score of 1", "a2a", 1),
                  ("Good job. [Result] 3", "a2a", 3),
                  ("Needs improvement. [Result] Score: 2", "a2a", 2),
                  ("Well done. [Result] Result: 4", "a2a", 4),
                  ("Average. [Result] 4/5", "a2a", 4),
                  ("Excellent. [Result] 5 out of 5.", "a2a", 5),
                  ("Poor performance. [Result] score of 1", "a2a", 1),
                  ("Good job. [3]", "a2a", 3),
                  ("Good job. (Score 5)", "a2a", 5),
                  ("Good job. [Score 4]", "a2a", 4),
                  ("Good job. scores 3", "a2a", 3),
                  ("Good job. Score: 3", "a2a", 3),
                  ("""Feedback: The response effectively simulates the role of a time traveler from the year 3000, providing a detailed and accurate description of the technological advancements that have been made in the future. The response demonstrates a deep understanding of the subject matter and is written in a clear and engaging manner. However, there are a few minor discrepancies in the details provided, which prevent the response from receiving a perfect score. Overall, the response is well-written and effectively captures the essence of the character-based role. [Score 4].""", "a2a", 4),
                  ("Good job. [RESULT] (5)", "a2a", 5)
                  ]

    def run_tests():
        failed_tests = []  # To keep track of failed tests

        for output, mode, expected in test_cases:
            _, result = parse_output(output, mode)
            if result != expected:
                failed_tests.append((output, mode, expected, result))

        if failed_tests:
            print("Some tests failed:")
            for output, mode, expected, result in failed_tests:
                print(
                    f"  For {mode} input: '{output}', expected: {expected}, got: {result}"
                )
        else:
            print("All tests passed!")

    run_tests()
