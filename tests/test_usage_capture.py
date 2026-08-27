"""Unit tests for usage_capture: begin/record/collect lifecycle and _to_dict."""

from types import SimpleNamespace

from basemode import usage_capture


def test_collect_without_begin_capture_returns_empty_list():
    # Fresh ContextVar default is None; collect must not blow up.
    assert usage_capture.collect() == []


def test_record_before_begin_capture_is_a_noop():
    # No active capture context: record should silently drop the usage.
    usage_capture.record({"prompt_tokens": 1})
    assert usage_capture.collect() == []


def test_record_none_is_ignored():
    usage_capture.begin_capture()
    usage_capture.record(None)
    assert usage_capture.collect() == []


def test_begin_capture_then_record_then_collect_dict_usage():
    usage_capture.begin_capture()
    usage_capture.record({"prompt_tokens": 10, "completion_tokens": 5})
    assert usage_capture.collect() == [{"prompt_tokens": 10, "completion_tokens": 5}]


def test_collect_accumulates_multiple_records_in_order():
    usage_capture.begin_capture()
    usage_capture.record({"prompt_tokens": 1})
    usage_capture.record({"prompt_tokens": 2})
    assert usage_capture.collect() == [
        {"prompt_tokens": 1},
        {"prompt_tokens": 2},
    ]


def test_begin_capture_resets_previous_events():
    usage_capture.begin_capture()
    usage_capture.record({"prompt_tokens": 1})
    usage_capture.begin_capture()
    assert usage_capture.collect() == []


def test_collect_is_idempotent_and_returns_a_copy():
    usage_capture.begin_capture()
    usage_capture.record({"prompt_tokens": 1})
    first = usage_capture.collect()
    first.append({"prompt_tokens": 999})
    second = usage_capture.collect()
    assert second == [{"prompt_tokens": 1}]


def test_record_with_model_dump_method():
    usage_capture.begin_capture()

    class Usage:
        def model_dump(self):
            return {"prompt_tokens": 7, "completion_tokens": 3}

    usage_capture.record(Usage())
    assert usage_capture.collect() == [{"prompt_tokens": 7, "completion_tokens": 3}]


def test_record_with_dict_method_fallback():
    usage_capture.begin_capture()

    class Usage:
        def dict(self):
            return {"prompt_tokens": 4}

    usage_capture.record(Usage())
    assert usage_capture.collect() == [{"prompt_tokens": 4}]


def test_record_with_model_dump_that_raises_falls_through():
    usage_capture.begin_capture()

    class Usage:
        def model_dump(self):
            raise RuntimeError("boom")

        prompt_tokens = 8
        completion_tokens = 2
        total_tokens = 10

    usage_capture.record(Usage())
    assert usage_capture.collect() == [
        {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
    ]


def test_record_with_dict_castable_object():
    usage_capture.begin_capture()
    # A mapping-like object without model_dump/dict, but castable via dict().
    usage_capture.record([("prompt_tokens", 6)])
    assert usage_capture.collect() == [{"prompt_tokens": 6}]


def test_record_with_plain_attribute_object_no_details():
    usage_capture.begin_capture()
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16)
    usage_capture.record(usage)
    assert usage_capture.collect() == [
        {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    ]


def test_record_with_plain_attribute_object_and_reasoning_details():
    usage_capture.begin_capture()
    details = SimpleNamespace(reasoning_tokens=9)
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        total_tokens=16,
        completion_tokens_details=details,
    )
    usage_capture.record(usage)
    assert usage_capture.collect() == [
        {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "completion_tokens_details": {"reasoning_tokens": 9},
        }
    ]


def test_record_with_empty_usage_object_records_nothing():
    usage_capture.begin_capture()

    class EmptyUsage:
        pass

    usage_capture.record(EmptyUsage())
    # _to_dict returns {} for an object with none of the known attributes,
    # and record() only appends when the dict is truthy.
    assert usage_capture.collect() == []
