import gzip
import re
from collections.abc import AsyncGenerator, AsyncIterable
from functools import lru_cache
from importlib import resources
from pathlib import Path

_LOOKBEHIND_CHARS = 80
_COMMIT_LAG_CHARS = 32

# Webster's Second International (the BSD `web2` list, 1934 copyright lapsed),
# shipped with the package. Boundary repair reads the dictionary for nearly
# every decision, so an empty one does not degrade gracefully — it inverts the
# outcome: with no known words, `_word_like` is false for everything, the
# prefix always looks mid-word, and "Seed beta" + " gamma" joins into
# "betagamma". That is exactly what happened wherever /usr/share/dict/words
# was absent, which is most Linux hosts and every stock CI runner. Bundling the
# list makes the behaviour the same on every machine.
_BUNDLED_DICT = "words.txt.gz"
_DICT_PATH = Path("/usr/share/dict/words")
_DICTIONARY: frozenset[str] | None = None


def _load_bundled_dictionary() -> frozenset[str] | None:
    try:
        raw = resources.files("basemode").joinpath("data", _BUNDLED_DICT).read_bytes()
    except Exception:
        return None
    try:
        return frozenset(gzip.decompress(raw).decode("utf-8").split())
    except Exception:
        return None


def _dictionary() -> frozenset[str]:
    """Known English words, lowercased.

    The packaged list is the source of truth so that healing is deterministic
    across machines; the system dictionary is only a fallback for an install
    whose package data went missing.
    """
    global _DICTIONARY
    if _DICTIONARY is None:
        bundled = _load_bundled_dictionary()
        if bundled is not None:
            _DICTIONARY = bundled
        elif _DICT_PATH.exists():
            _DICTIONARY = frozenset(_DICT_PATH.read_text().lower().split())
        else:
            _DICTIONARY = frozenset()
    return _DICTIONARY


def _is_word(s: str) -> bool:
    return s.lower() in _dictionary()


_VOCAB_RE = re.compile(r"[A-Za-z]+")


@lru_cache(maxsize=4)
def _document_vocabulary(prefix: str) -> frozenset[str]:
    """Words that already stand complete in the document.

    The system dictionary is missing most inflected forms and knows nothing of
    invented or proper nouns, which is exactly the vocabulary loom documents are
    full of. The text so far is a better dictionary for its own coinages.

    The final token is dropped: when generation resumes mid-word that trailing
    fragment is the one string we must not treat as an established word.
    """
    matches = list(_VOCAB_RE.finditer(prefix))
    if matches and matches[-1].end() == len(prefix):
        matches.pop()
    return frozenset(match.group(0).lower() for match in matches)


def _stem_candidates(word: str) -> list[str]:
    """Plausible stems of an inflected form, for dictionary lookup.

    ``/usr/share/dict/words`` lists ``flank`` but not ``flanks``, ``count`` but
    not ``counted``. Without this, ordinary plurals and past tenses look like
    word fragments and get glued onto the preceding word.
    """
    stems: list[str] = []

    def add(stem: str) -> None:
        if len(stem) >= 3:
            stems.append(stem)

    if word.endswith("ies"):
        add(word[:-3] + "y")
    if word.endswith("es"):
        add(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        add(word[:-1])
    if word.endswith("ed"):
        add(word[:-2])
        add(word[:-1])
        if len(word) > 4 and word[-3] == word[-4]:
            add(word[:-3])
    if word.endswith("ing"):
        add(word[:-3])
        add(word[:-3] + "e")
        if len(word) > 5 and word[-4] == word[-5]:
            add(word[:-4])
    if word.endswith("ly"):
        add(word[:-2])
    return stems


# The only English words spelled with one letter. A lone letter opening a
# continuation is otherwise the tail of the word before it: "an" + " d".
_STANDALONE_LETTERS = {"a", "i", "o"}

# The system dictionary lists plenty of two-letter entries that never appear as
# words in prose ("en", "ad", "ne"), and treating them as words blocks obvious
# repairs like "be" + " en". Interjections are kept so that dialogue survives:
# "a moth" + " er, no" must not become "mother, no".
_TWO_LETTER_WORDS = {
    "ah",
    "am",
    "an",
    "as",
    "at",
    "ax",
    "be",
    "by",
    "do",
    "eh",
    "er",
    "ex",
    "go",
    "ha",
    "he",
    "hi",
    "hm",
    "ho",
    "id",
    "if",
    "in",
    "is",
    "it",
    "lo",
    "ma",
    "me",
    "mm",
    "my",
    "no",
    "of",
    "oh",
    "ok",
    "on",
    "or",
    "ow",
    "ox",
    "pa",
    "pi",
    "so",
    "to",
    "uh",
    "um",
    "up",
    "us",
    "we",
    "ye",
    "yo",
}


def _can_stand_alone(word: str, vocab: frozenset[str] = frozenset()) -> bool:
    """True if ``word`` can open a continuation as a word in its own right."""
    lowered = word.lower()
    if lowered in vocab:
        return True
    if len(word) == 1:
        return lowered in _STANDALONE_LETTERS
    if len(word) == 2:
        return lowered in _TWO_LETTER_WORDS
    return _word_like(word, vocab)


def _word_like(word: str, vocab: frozenset[str] = frozenset()) -> bool:
    """True if ``word`` plausibly stands on its own rather than being a fragment."""
    lowered = word.lower()
    if lowered in _COMMON_SHORT_WORDS or lowered in vocab or _is_word(lowered):
        return True
    return any(stem in vocab or _is_word(stem) for stem in _stem_candidates(lowered))


_COMPOUND_RE = re.compile(r"\b([A-Za-z]{2,}) (?=([A-Za-z]{2,})\b)")
_PREFIX_WORD_RE = re.compile(r"([A-Za-z]+)$")
_TRAILING_FRAGMENT_RE = re.compile(r"(?:^|(?<=\s))([A-Za-z]{1,3})$")
_PREFIX_HYPHEN_FRAGMENT_RE = re.compile(r"([A-Za-z]+-[A-Za-z]{0,2})$")
_LEADING_WORD_RE = re.compile(r"^ ([A-Za-z]+)(\b|(?=[^A-Za-z]))")
# A run of whitespace containing at least one newline, then a word.
_LEADING_NEWLINE_WORD_RE = re.compile(r"^([ \t]*\n\s*)([A-Za-z]+)")
_DANGLING_HYPHENATED_TAIL_RE = re.compile(r"\b[A-Za-z]{2,}-(?:[A-Za-z]{0,2})$")


def normalize_prefix(prefix: str) -> str:
    """Ensure prefix ends with exactly one space for the model input.

    Chat models respond without a leading space, so we strip trailing whitespace
    and add exactly one space. This makes the model output tokens that join
    correctly when we prepend a space to the first token if needed.

    Not applied to completion/prefill strategies — they handle boundaries natively.
    """
    return prefix.rstrip() + " "


def rewind_prefix_to_word_boundary(prefix: str) -> tuple[str, str]:
    """Return a generation prefix and the trailing token removed from it.

    Chat strategies cannot show a model where a word ends without also handing
    it a trailing space, which is what creates split words at the boundary. The
    way out is to not send the last short token at all: generate from
    ``"twas brilig and the "``, and if the model comes back with ``"slithy"``
    the caller strips the ``"sli"`` it already has and appends ``"thy"``. The
    join is then exact rather than guessed.

    Only a short whole token at a whitespace boundary is rewound — long tails
    are unlikely to be reproduced, and the caller pays for a second request
    whenever the model declines to re-emit the fragment. Complete words are
    rewound too: ``"counted, a"`` is precisely the case where the model wants to
    write ``"and"`` and the injected space lands inside it.
    """
    match = _TRAILING_FRAGMENT_RE.search(prefix)
    if not match:
        return prefix, ""

    return prefix[: match.start(1)], match.group(1)


async def probe_rewind_overlap(
    tokens: AsyncIterable[str],
    fragment: str,
) -> tuple[bool, str]:
    """Decide whether a rewound generation re-emitted the trailing fragment.

    Returns ``(matched, head)``. When matched, ``head`` is the buffered text
    with the duplicated fragment removed and the caller can stream the rest.
    When not matched the generation was conditioned on a prefix the reader never
    sees, so ``head`` must be discarded rather than shown: the caller retries
    without the rewind.
    """
    if not fragment:
        return True, ""

    buffer = ""
    target = fragment.lower()
    max_probe = len(fragment) + 1

    async for token in tokens:
        buffer += token
        if len(buffer) >= max_probe:
            break

    probe = buffer.lower()
    if probe.startswith(target):
        return True, buffer[len(fragment) :]
    if probe.startswith(" " + target):
        return True, buffer[len(fragment) + 1 :]
    return False, buffer


def needs_leading_space(prefix: str, first_token: str) -> bool:
    """Return True if a space must be injected between prefix and first_token.

    After sending normalize_prefix(prefix) to the model, the model outputs
    first_token without a leading space. If the original prefix didn't end
    with whitespace, the space was consumed in the model input and must be
    restored so that prefix + tokens is correct text.
    """
    return (
        bool(prefix)
        and not prefix[-1].isspace()
        and bool(first_token)
        and not first_token[0].isspace()
    )


def _looks_line_oriented(text: str) -> bool:
    lines = [line for line in text.rstrip().splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    recent = lines[-4:]
    avg_len = sum(len(line.strip()) for line in recent) / len(recent)
    punctuation_endings = sum(
        line.rstrip().endswith((".", "!", "?", ":", ";")) for line in recent
    )
    markdown_starts = sum(
        line.lstrip().startswith(("#", ">", "-", "*", "+", "```")) for line in recent
    )

    return markdown_starts > 0 or (avg_len < 48 and punctuation_endings <= 1)


def _should_collapse_single_newline(
    prefix: str, prev_char: str, next_char: str
) -> bool:
    if _looks_line_oriented(prefix):
        return False
    if not prev_char or not next_char:
        return False
    if prev_char.isspace() or next_char.isspace():
        return False
    if next_char in "#>-*+`|":
        return False
    return True


_JOINABLE_COMPOUNDS = {
    "anyone",
    "anything",
    "anywhere",
    "cowardice",
    "everyone",
    "everything",
    "everywhere",
    "herself",
    "himself",
    "inside",
    "itself",
    "myself",
    "nothing",
    "nowhere",
    "ourselves",
    "outside",
    "someone",
    "something",
    "somewhere",
    "themselves",
    "yourself",
    "yourselves",
}

_COMMON_SHORT_WORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "did",
    "do",
    "for",
    "had",
    "has",
    "he",
    "her",
    "him",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "may",
    "me",
    "not",
    "now",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "own",
    "she",
    "so",
    "the",
    "to",
    "too",
    "two",
    "up",
    "us",
    "was",
    "we",
    "who",
    "why",
    "you",
}


# Determiners whose split reading is a real noun phrase: "every one of them",
# "laid out side by side".
_COMPOUND_NP_HEADS = {"every", "any", "some", "no", "out", "in"}
_NP_TAIL_RE = re.compile(r"\s+(?:of|by)\b")

# "in"/"out" are prepositions far more often than they are the head of a split
# compound, so they are only joined where a determiner makes the pair a noun:
# "made the out side feel theoretical" but not "cars in side streets".
_PREPOSITION_HEADS = {"in", "out"}
_DETERMINER_RE = re.compile(
    r"\b(?:the|a|an|this|that|these|those|its|his|her|their|our|my|your)\s+$",
    re.IGNORECASE,
)


def _blocks_compound_join(left: str, before: str, after: str) -> bool:
    """True if the surrounding text shows the words were meant to stay apart."""
    if after.startswith("-"):
        # "her self-esteem" is not "herself-esteem"; "any one-way trip" stands.
        return True
    lowered = left.lower()
    if lowered in _COMPOUND_NP_HEADS and _NP_TAIL_RE.match(after):
        return True
    if lowered in _PREPOSITION_HEADS and not _DETERMINER_RE.search(before):
        return True
    return False


def _join_split_compounds(text: str, *, before: str = "", protect_tail: int = 0) -> str:
    """Join high-confidence compounds in the mutable stream tail.

    ``before`` is the text immediately preceding ``text`` (the prefix tail, or
    what has already been emitted) and ``protect_tail`` leaves the last N
    characters undecided, so a pair at either edge of the buffer is only joined
    once ``_blocks_compound_join`` can see the words around it.
    """
    limit = len(text) - protect_tail

    def replace(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        end = match.end() + len(right)
        if end > limit:
            return match.group(0)
        if (left + right).lower() not in _JOINABLE_COMPOUNDS:
            return match.group(0)
        if _blocks_compound_join(left, before + text[: match.start()], text[end:]):
            return match.group(0)
        return left

    return _COMPOUND_RE.sub(replace, text)


_SPACE_PUNCT_RE = re.compile(r"(?<=\w) ([,\.;:!?])")
_SPACE_CONTRACTION_RE = re.compile(r"(?<=\w) ('(?:s|t|re|ve|ll|d|m))\b", re.IGNORECASE)
_LEADING_PUNCT_RE = re.compile(r"^ ([,\.;:!?])")
_LEADING_CONTRACTION_RE = re.compile(r"^ ('(?:s|t|re|ve|ll|d|m))\b", re.IGNORECASE)


def _fix_space_before_punctuation(text: str) -> str:
    text = _SPACE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_CONTRACTION_RE.sub(r"\1", text)
    return text


def _repair_newline_split_word(prefix: str, text: str, protect_tail: int) -> str | None:
    """Drop a line break a model opened with in the middle of a word.

    A continuation may begin with a newline for perfectly good reasons — a new
    paragraph, a heading, a scene break — but not when the prefix stops
    mid-word: nothing legitimate starts a paragraph inside "thermodynamic".
    Seen in the wild as ``"...work in the thermod"`` continued by
    ``"\n\nynamic sense is only possible"``.

    The evidence required is the same as for the space case, so a real
    paragraph break after a complete word is never touched: the prefix must
    end on a fragment that is not a word, and the fragment plus the
    continuation's first word must be one. Returns None when the rule does
    not apply, leaving the remaining repairs to decide.
    """
    prefix_match = _PREFIX_WORD_RE.search(prefix)
    if not prefix_match or (prefix and prefix[-1].isspace()):
        return None
    match = _LEADING_NEWLINE_WORD_RE.match(text)
    if not match:
        return None
    left, right = prefix_match.group(1), match.group(2)
    # Too little of the word has arrived to judge it; the caller runs again
    # when the stream ends, with nothing held back.
    if len(text) - match.end(2) < protect_tail:
        return None
    vocab = _document_vocabulary(prefix)
    if _word_like(left, vocab) or not _word_like(left + right, vocab):
        return None
    return text[match.end(1) :]


def _repair_prefix_boundary(prefix: str, text: str, *, protect_tail: int = 0) -> str:
    """Undo a space that landed inside a word straddling the generation boundary.

    Chat strategies must send the prefix with a trailing space and re-insert one
    before the first token, which destroys the distinction between a model
    starting a new word and a model finishing the prefix's last one. This puts
    the boundary back together when the evidence says the two halves are one
    word.
    """
    rejoined = _repair_newline_split_word(prefix, text, protect_tail)
    if rejoined is not None:
        return rejoined

    prefix_match = _PREFIX_WORD_RE.search(prefix)

    # Space injected before punctuation or contraction at the boundary
    if prefix_match:
        if _LEADING_PUNCT_RE.match(text) or _LEADING_CONTRACTION_RE.match(text):
            return text[1:]

    text_match = _LEADING_WORD_RE.match(text)
    hyphen_match = _PREFIX_HYPHEN_FRAGMENT_RE.search(prefix)
    if hyphen_match and text_match:
        return text[1:]

    if not prefix_match or not text_match:
        return text

    left = prefix_match.group(1)
    right = text_match.group(1)
    joined = left + right
    after = text[text_match.end(1) :]

    # Whitelist: high-confidence compounds where both halves are real words
    if joined.lower() in _JOINABLE_COMPOUNDS:
        if len(after) < protect_tail:
            return text
        return text if _blocks_compound_join(left, prefix, after) else text[1:]

    if after.startswith("-"):
        # "a" + " re-evaluation": the right half heads a hyphenated compound.
        return text

    if right[:1].isupper() and left[-1:].islower():
        # Models do not capitalise mid-word, so this is a proper noun starting a
        # new word: "his brother" + " Ed", not "brothered".
        return text

    vocab = _document_vocabulary(prefix)
    right_alone = _can_stand_alone(right, vocab)
    merged_is_word = _word_like(joined, vocab)

    if not _word_like(left, vocab):
        # The prefix already ends mid-word, so the fragment completes it unless
        # it is plainly a word of its own ("recalculat" + " ed", "fl" + " anks").
        return text[1:] if merged_is_word or not right_alone else text

    # The prefix ends on a real word, so only join when the fragment cannot
    # stand alone and the merged form is itself a word: "a" + " nd" -> "and".
    return text[1:] if merged_is_word and not right_alone else text


_WORD_RE = re.compile(r"[A-Za-z']+")
_TRAILING_PUNCT_RE = re.compile(r"[,.;:!?]+$")

# Repetition of these alone is ordinary prose ("he said he would"), not a
# model echoing what it was just shown, so a run of only such words never
# counts as evidence of a restated prefix.
_REPETITION_NOISE_WORDS = _COMMON_SHORT_WORDS | _TWO_LETTER_WORDS | _STANDALONE_LETTERS


def _strip_repeated_leading_words(prefix: str, text: str, max_words: int = 3) -> str:
    """Drop leading words that exactly restate the prefix's trailing words.

    Some models echo the word(s) they were just shown before continuing:
    prefix ``"...called Langton "`` followed by completion ``"Langton found
    he called..."``. Matching is case-insensitive, tries the longest run
    first (so a repeated phrase is trimmed once rather than word by word),
    and requires at least one non-trivial word in the run — pure function
    words ("the", "he", "and") repeat naturally in prose and are not evidence
    of an echo.
    """
    stripped = text.lstrip(" ")
    leading_space = text[: len(text) - len(stripped)]
    prefix_words = _WORD_RE.findall(prefix.rstrip())
    if not prefix_words:
        return text

    text_words = stripped.split(" ") if stripped else []
    limit = min(max_words, len(prefix_words), len(text_words))

    for n in range(limit, 0, -1):
        tail = [w.lower() for w in prefix_words[-n:]]
        head = [_TRAILING_PUNCT_RE.sub("", w).lower() for w in text_words[:n]]
        if tail != head:
            continue
        if all(w in _REPETITION_NOISE_WORDS for w in tail):
            continue
        remainder = " ".join(text_words[n:])
        return leading_space + remainder if remainder else ""

    return text


def _trim_dangling_short_tail(text: str, vocab: frozenset[str] = frozenset()) -> str:
    match = re.search(r"\b([A-Za-z]{1,3})$", text)
    if not match:
        return text
    if _word_like(match.group(1), vocab):
        return text
    return text[: match.start(1)].rstrip()


def normalize_completion_segment(prefix: str, completion: str) -> str:
    """Repair a generated segment before persisting it as a child node.

    Stream normalization can repair most boundaries while tokens arrive, but
    persistence is the last line of defense: no node should be born with an
    avoidable leading split-word fragment or an obvious dangling hyphenated tail
    caused by a token limit.
    """
    completion = _repair_prefix_boundary(prefix, completion)
    completion = _strip_repeated_leading_words(prefix, completion)
    completion = _DANGLING_HYPHENATED_TAIL_RE.sub("", completion).rstrip()
    return _trim_dangling_short_tail(completion, _document_vocabulary(prefix))


async def normalize_stream_newlines(
    prefix: str,
    tokens: AsyncIterable[str],
) -> AsyncGenerator[str, None]:
    """Collapse likely hard-wrapped prose newlines and repair split compounds.

    The final few characters are held back briefly so the next token can repair
    boundaries such as ``coward ice`` -> ``cowardice`` before text is committed
    to the caller.
    """
    prev_char = prefix[-1] if prefix else ""
    pending_newlines = 0
    pending_text = ""
    at_boundary = True
    recent = prefix[-_COMMIT_LAG_CHARS:]

    async for token in tokens:
        out: list[str] = []
        for char in token:
            if char == "\n":
                pending_newlines += 1
                continue

            if pending_newlines:
                if pending_newlines == 1 and _should_collapse_single_newline(
                    prefix, prev_char, char
                ):
                    if prev_char != " ":
                        out.append(" ")
                        prev_char = " "
                else:
                    out.append("\n" * pending_newlines)
                    prev_char = "\n"
                pending_newlines = 0

            out.append(char)
            prev_char = char

        if out:
            pending_text += "".join(out)
            if at_boundary:
                pending_text = _repair_prefix_boundary(
                    prefix, pending_text, protect_tail=_COMMIT_LAG_CHARS
                )
                pending_text = _strip_repeated_leading_words(prefix, pending_text)
            pending_text = _join_split_compounds(
                pending_text, before=recent, protect_tail=_COMMIT_LAG_CHARS
            )
            pending_text = _fix_space_before_punctuation(pending_text)
            if len(pending_text) > _LOOKBEHIND_CHARS:
                emit_len = len(pending_text) - _COMMIT_LAG_CHARS
                emitted = pending_text[:emit_len]
                yield emitted
                recent = (recent + emitted)[-_COMMIT_LAG_CHARS:]
                pending_text = pending_text[emit_len:]
                at_boundary = False

    if pending_newlines:
        pending_text += "\n" * pending_newlines
    if at_boundary:
        pending_text = _repair_prefix_boundary(prefix, pending_text)
        pending_text = _strip_repeated_leading_words(prefix, pending_text)
    pending_text = _join_split_compounds(pending_text, before=recent)
    pending_text = _fix_space_before_punctuation(pending_text)
    if pending_text:
        yield pending_text
