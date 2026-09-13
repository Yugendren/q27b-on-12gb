| build | L40S sm_89 (k/164) | RTX 3060 sm_86 (k/n) | 3060 95% CI | delta |
|---|---|---|---|---|
| IQ3_XXS | 154/164 = 93.9% | 152/164 = 92.7% | 87.6%-95.8% | -1.2 pt |
| Q2_K_XL | 154/164 = 93.9% | 153/164 = 93.3% | 88.4%-96.2% | -0.6 pt |
| IQ2_XXS | 132/164 = 80.5% | 128/164 = 78.0% | 71.1%-83.7% | -2.4 pt |
| Q8_0 | 153/164 = 93.3% | 52/54 = 96.3% | 87.5%-99.0% | +3.0 pt |

### Greedy determinism across platforms

Fraction of paired tasks where the two GPUs emitted the SAME number of completion tokens. Greedy decoding is deterministic given identical arithmetic, so this is a direct proxy for kernel-level numerical agreement -- independent of whether the answer was right.

| build | identical token count | mean |L40S - 3060| tokens |
|---|---|---|---|
| IQ3_XXS | 150/164 = 91.5% | 7.2 |
| Q2_K_XL | 136/164 = 82.9% | 6.5 |
| IQ2_XXS | 110/164 = 67.1% | 28.9 |
| Q8_0 | 54/54 = 100.0% | 0.0 |

### Paired McNemar, SAME build across platforms (key = task_id, N=1 greedy)

| build | n paired | both pass | b (L40S pass / 3060 fail) | c (L40S fail / 3060 pass) | neither | p (exact) |
|---|---|---|---|---|---|---|
| IQ3_XXS | 164 | 152 | 2 | 0 | 10 | 0.5 |
| Q2_K_XL | 164 | 153 | 1 | 0 | 10 | 1 |
| IQ2_XXS | 164 | 122 | 10 | 6 | 26 | 0.4545 |
| Q8_0 | 54 | 52 | 0 | 0 | 2 | 1 |

### Flipped tasks (L40S pass -> 3060 fail), with the 3060 failure mode

**IQ3_XXS** — 2 regressions, 0 recoveries

| task | 3060 error | 3060 completion_tokens | L40S completion_tokens |
|---|---|---|---|
| HumanEval/32 | `AssertionError` | 625 | 790 |
| HumanEval/91 | `AssertionError` | 169 | 239 |

**Q2_K_XL** — 1 regressions, 0 recoveries

| task | 3060 error | 3060 completion_tokens | L40S completion_tokens |
|---|---|---|---|
| HumanEval/83 | `AssertionError` | 336 | 817 |

**IQ2_XXS** — 10 regressions, 6 recoveries

| task | 3060 error | 3060 completion_tokens | L40S completion_tokens |
|---|---|---|---|
| HumanEval/11 | `AssertionError` | 1 | 136 |
| HumanEval/13 | `AssertionError` | 1 | 95 |
| HumanEval/40 | `SyntaxError` | 1024 | 336 |
| HumanEval/44 | `AssertionError` | 1 | 163 |
| HumanEval/71 | `AssertionError` | 206 | 205 |
| HumanEval/75 | `AssertionError` | 223 | 227 |
| HumanEval/94 | `SyntaxError` | 1024 | 476 |
| HumanEval/102 | `AssertionError` | 1 | 145 |
| HumanEval/126 | `AssertionError` | 303 | 303 |
| HumanEval/138 | `AssertionError` | 94 | 104 |

recoveries: HumanEval/1, HumanEval/10, HumanEval/46, HumanEval/91, HumanEval/124, HumanEval/156

**Q8_0** — 0 regressions, 0 recoveries

### Side-by-side generations on flipped tasks (bug-report evidence)

Same GGUF, same prompt, same greedy settings, same harness file; only the GPU differs.

#### IQ3_XXS — HumanEval/91

L40S sm_89 (PASS, 239 tok):

```
```python
def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!'.
   
    For example:
    >>> is_bored("Hello world")
    0
    >>> is_bored("The sky is blue. The sun is shining. I love this weather")
    1
    """
    import re
    # Split the string into sentences based on '.', '?' or '!'
    sentences = re.split(r'[.?!]', S)
    count = 0
    for sentence in sentences:
        # Strip whitespace and check if the sentence starts with "I"
        stripped = sentence.strip()
        if stripped.startswith("I"):
            # Ensure "I" is a standalone word (followed by space or end of string)
            if len(stripped) == 1 or stripped[1] == ' ':
                count += 1
    return count
```
```

RTX 3060 sm_86 (FAIL: AssertionError, 169 tok):

```
```python
def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!'.
   
    For example:
    >>> is_bored("Hello world")
    0
    >>> is_bored("The sky is blue. The sun is shining. I love this weather")
    1
    """
    import re
    sentences = re.split(r'[.?!]', S)
    count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence.startswith("I"):
            count += 1
    return count
```
```

#### IQ2_XXS — HumanEval/11 (L40S text from reliability_IQ2_XXS_nospec_oldbuild.jsonl)

L40S sm_89 (PASS, 136 tok):

```
```python
from typing import List


def string_xor(a: str, b: str) -> str:
    """ Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """
    result = []
    for i in range(len(a)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
    return ''.join(result)
```
```

RTX 3060 sm_86 (FAIL: AssertionError, 1 tok):

```

```

#### IQ2_XXS — HumanEval/13 (L40S text from reliability_IQ2_XXS_nospec_oldbuild.jsonl)

L40S sm_89 (PASS, 95 tok):

```
```python
def greatest_common_divisor(a: int, b: int) -> int:
    """ Return a greatest common divisor of two integers a and b
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """
    while b:
        a, b = b, a % b
    return a
```
```

RTX 3060 sm_86 (FAIL: AssertionError, 1 tok):

```

```

### Failure-mode census (3060 vs L40S), attempt 1

| build | platform | fails | error-type counts | fence_unclosed |
|---|---|---|---|---|
| IQ3_XXS | L40S sm_89 | 10 | {'NameError': 2, 'AssertionError': 8} | 2 |
| IQ3_XXS | RTX 3060 sm_86 | 12 | {'AssertionError': 9, 'NameError': 2, 'Traceback (most recent call la': 1} | 2 |
| Q2_K_XL | L40S sm_89 | 10 | {'NameError': 2, 'AssertionError': 6, 'Error': 1, 'SyntaxError': 1} | 0 |
| Q2_K_XL | RTX 3060 sm_86 | 11 | {'NameError': 2, 'AssertionError': 8, 'Error': 1} | 1 |
| IQ2_XXS | L40S sm_89 | 32 | {'NameError': 3, 'AssertionError': 15, 'IndexError': 2, 'SyntaxError': 7, 'ModuleNotFoundError': 1, 'File "/tmp/humaneval_4simmb3j.': 1, 'File "/tmp/humaneval_qmi61sj2.': 1, 'Error': 1, 'TypeError': 1} | 0 |
| IQ2_XXS | RTX 3060 sm_86 | 36 | {'AssertionError': 22, 'IndexError': 2, 'NameError': 3, 'SyntaxError': 6, 'File "/tmp/humaneval_ppw5c08g.': 1, 'File "/tmp/humaneval__2z_a5jp.': 1, 'Error': 1} | 6 |
| Q8_0 | L40S sm_89 | 11 | {'NameError': 2, 'AssertionError': 5, 'Error': 1, 'SyntaxError': 3} | 0 |
| Q8_0 | RTX 3060 sm_86 | 2 | {'NameError': 2} | 0 |
