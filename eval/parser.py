import re


def _parse_output_absolute(output):
    # First, try to find "out of 5" or "/5" and extract the score before it
    out_of_5_index = output.lower().find("out of 5")
    if out_of_5_index == -1:  # If "out of 5" is not found, check for "/5"
        out_of_5_index = output.find("/5")

    if out_of_5_index != -1:
        # Find the score before "out of 5" or "/5"
        score_part = output[:out_of_5_index].strip()
        # Extract the last number before "out of 5" or "/5"
        match = re.search(r'(\d+)\s*$', score_part)
        if match:
            result = int(match.group(1))
            if 1 <= result <= 5:
                return output, result

    # If "out of 5" or "/5" is not found, fall back to the regular expression
    pattern = r"""
        (?:                        # Start of non-capturing group
            \[RESULT\]|\[SCORE\]|   # Match [RESULT] or [SCORE]
            Score:?|score:?|        # Match Score: or score:
            Result:?|\[Result\]:?|  # Match Result: or [Result]:
            score\s+of|             # Match "score of"
            \[|\(                   # Match opening bracket or parenthesis
        )                           # End of non-capturing group
        \s*                         # Allow any whitespace
        (\d+)                       # Capture the digit(s)
        (?:                         # Start of non-capturing group
            (?:\)|\]|\s|$)|         # Allow closing brackets, whitespace, or end of string
            (?:/\s*5|               # Allow /5 with optional whitespace
               \s*out\s*of\s*5)     # or "out of 5" with flexible whitespace
        )?                          # End of non-capturing group
        (?:\s*\.*\s*)               # Allow optional period and whitespace
        (?:.*)?                     # Allow any additional text after the score
    """
    match = re.search(pattern, output, re.IGNORECASE | re.VERBOSE)

    if match:
        result = int(match.group(1))
        if 1 <= result <= 5:  # Ensure the result is within the valid range
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
                  ("Good job. score: 3", "a2a", 3),
                  ("Good job. Score: 3", "a2a", 3),
                  (""" Feedback:

The response provides a good effort in simulating the role of a character in a post-apocalyptic world. The writer does a good job of conveying the harsh realities of surviving in a world that has been ravaged by an unspecified cataclysmic event. The description of the protagonist's daily struggles, such as scavenging for food and defending against mutated creatures and raiders, is vivid and immersive.

However, there are some areas where the response could be improved. For example, the protagonist's motivations and emotions could be more fully fleshed out. While the writer mentions that the protagonist is grateful to be alive and has formed a sense of camaraderie with their allies, there is little depth or nuance to these emotions. Additionally, the response could benefit from more specific historical or cultural details that would help to ground the story in a particular time and place.

Overall, the response scores a 3 out of 5 on the score rubric. While it effectively simulates the role of a character in a post-apocalyptic world, there is room for improvement in terms of character development and historical accuracy.""", "a2a", 3),
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
